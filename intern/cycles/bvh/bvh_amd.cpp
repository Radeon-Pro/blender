#include "render/mesh.h"
#include "render/object.h"
#include "util/util_foreach.h"
#include "util/util_logging.h"
#include "util/util_progress.h"
#include "util/util_transform.h"

#include "bvh/bvh_node.h"
#include "bvh/bvh_amd.h"
#include "render/hair.h"

#include "bvh/bvh2.h"
#include "render/curves.h"


#include "bvh/bvh_binning.h"
#include "bvh/bvh_build.h"
//#include "util/util_time.h"
//#include "util/util_progress.h"


CCL_NAMESPACE_BEGIN
static thread_mutex bbox_mutex;


struct EmbreeNode {
  int nodeType;
  uint32_t visibility;
  bool is_leaf()
  {
    return (nodeType == 0 || nodeType == 6);
  }
};

struct EmbreeCompactNode : public EmbreeNode {

  std::array<EmbreeNode *, 4> pointer{};

  std::array<BoundBox, 4> boundingBoxes{};

  EmbreeCompactNode()
  {
    nodeType = 5;
    visibility = 0;
  }

  static void *create(RTCThreadLocalAllocator alloc, unsigned int numChildren, void *userPtr)
  {

    assert(numChildren <= 4);
    assert(numChildren >= 2);
    void *ptr = rtcThreadLocalAlloc(alloc, sizeof(EmbreeCompactNode), 16);
    return (void *)new (ptr) EmbreeCompactNode;
  }

  static void setChildren(void *nodePtr, void **childPtr, unsigned int numChildren, void *userPtr)
  {

    assert(numChildren <= 4);
    assert(numChildren >= 2);
    for (size_t i = 0; i < 4; i++) {
      if (i < numChildren) {
        EmbreeNode *child = (EmbreeNode *)childPtr[i];

        ((EmbreeCompactNode *)nodePtr)->pointer[i] = child;
        if (child != nullptr)
          ((EmbreeCompactNode *)nodePtr)->visibility |= child->visibility;
      }
      else {
        ((EmbreeCompactNode *)nodePtr)->pointer[i] = nullptr;
      }
    }
  }

  static void setBounds(void *nodePtr,
                        const RTCBounds **bounds,
                        unsigned int numChildren,
                        void *userPtr)
  {

    assert(numChildren <= 4);
    assert(numChildren >= 2);
    for (size_t i = 0; i < numChildren; i++) {
      float3 min;
      min.x = bounds[i]->lower_x;
      min.y = bounds[i]->lower_y;
      min.z = bounds[i]->lower_z;
      ((EmbreeCompactNode *)nodePtr)->boundingBoxes[i].min = min;
      float3 max;
      max.x = bounds[i]->upper_x;
      max.y = bounds[i]->upper_y;
      max.z = bounds[i]->upper_z;
      ((EmbreeCompactNode *)nodePtr)->boundingBoxes[i].max = max;
    }
  }
};

struct EmbreeCompactTriangleNode : public EmbreeNode {
  uint32_t prim_id;

  EmbreeCompactTriangleNode(const RTCBuildPrimitive *prims)
  {
    prim_id = prims->primID;
    visibility = prims->geomID & (~PATH_RAY_NODE_UNALIGNED);
    nodeType = 0;
  }

  static void *create(RTCThreadLocalAllocator alloc,
                      const RTCBuildPrimitive *prims,
                      size_t numPrims,
                      void *userPtr)
  {
    assert(numPrims == 1);
    void *ptr = rtcThreadLocalAlloc(alloc, sizeof(EmbreeCompactTriangleNode), 16);
    return (void *)new (ptr) EmbreeCompactTriangleNode(prims);
  }
};

struct EmbreeCompactInstanceNode : public EmbreeNode {

  uint32_t primId;
  int object_id;

  EmbreeCompactInstanceNode(const RTCBuildPrimitive *prims, void *userPtr)
  {
    primId = prims->primID;
    object_id = prims->geomID >> 1;
    nodeType = (prims->geomID & 1) ? 6 : 0;
    if (nodeType == 0) {
      Object **objects = ((Object **)(userPtr));
      visibility = objects[object_id]->visibility_for_tracing();
    }
    else
      visibility = object_id;

    visibility &= (~PATH_RAY_NODE_UNALIGNED);
  }

  static void *create(RTCThreadLocalAllocator alloc,
                      const RTCBuildPrimitive *prims,
                      size_t numPrims,
                      void *userPtr)
  {
    assert(numPrims == 1);
    void *ptr = rtcThreadLocalAlloc(alloc, sizeof(EmbreeCompactInstanceNode), 16);
    return (void *)new (ptr) EmbreeCompactInstanceNode(prims, userPtr);
  }
};

class BVHBuild_Custom : public BVHBuild {

 public:
  BVHBuild_Custom(const vector<Object *> &objects,
                  array<int> &prim_type,
                  array<int> &prim_index,
                  array<int> &prim_object,
                  array<float2> &prim_time,
                  const BVHParams &params,
                  Progress &progress)
      : BVHBuild(objects, prim_type, prim_index, prim_object, prim_time, params, progress)
  {
  }
  BVHNode *run()
  {
    BVHRange root = BVHRange(bounds, center, 0, references.size());

    prim_type.resize(references.size());
    prim_index.resize(references.size());
    prim_object.resize(references.size());

    need_prim_time = params.num_motion_curve_steps > 0 || params.num_motion_triangle_steps > 0;

    if (need_prim_time)
      prim_time.resize(references.size());
    else
      prim_time.resize(0);

    if (params.top_level) {
      params.use_spatial_split = false;
    }

    spatial_min_overlap = root.bounds().safe_area() * params.spatial_split_alpha;
    spatial_free_index = 0;

    double build_start_time;
    build_start_time = progress_start_time = time_dt();
    progress_count = 0;
    progress_total = references.size();
    progress_original_total = progress_total;

    BVHNode *rootnode;

    if (params.use_spatial_split) {
      BVHSpatialStorage *local_storage = &spatial_storage.local();
      rootnode = build_node(root, references, 0, local_storage);
      task_pool.wait_work();
    }
    else {
      BVHObjectBinning rootbin(root, (references.size()) ? &references[0] : NULL);
      rootnode = build_node(rootbin, 0);
      task_pool.wait_work();
    }

    spatial_storage.clear();

    if (rootnode) {
      if (progress.get_cancel()) {
        rootnode->deleteSubtree();
        rootnode = NULL;
      }
      else {
        rootnode->update_visibility();
        rootnode->update_time();
      }
    }

    return rootnode;
  }
  void set_reference(std::vector<BVHReference> &reference, BoundBox in_bounds, BoundBox in_center)
  {
    references.resize(reference.size());
    std::copy(reference.begin(), reference.end(), references.begin());

    bounds = in_bounds;
    center = in_center;
  }

#ifdef SINGLE_LEVEL
  BVHNode *create_leaf_node(const BVHRange &range, const vector<BVHReference> &references)
  {
    const BVHReference &ref = references[range.start()];
    LeafNode *leaf_node = NULL;
    size_t start_index = 0;

    leaf_node = new LeafNode(ref.bounds(),
                             objects[ref.prim_object()]->visibility_for_tracing(),
                             start_index,
                             start_index + 1);
    leaf_node->lo = (ref.prim_index() == -1) ? range.start() : ref.prim_index();
    leaf_node->hi = ref.prim_object();

    float time_from = 1.0f, time_to = 0.0f;

    time_from = min(time_from, ref.time_from());
    time_to = max(time_to, ref.time_to());

    leaf_node->time_from = time_from;
    leaf_node->time_to = time_to;

    return leaf_node;
  }
#endif

 private:
  BoundBox bounds;
  BoundBox center;
};

BVHAMD::BVHAMD(const BVHParams &params,
               const vector<Geometry *> &geometry,
               const vector<Object *> &objects)
              //, const Device *device)
    : BVH(params, geometry, objects),
      root_bounds(BoundBox::empty),
      center(BoundBox::empty),
      is_embree(true)
{
  is_embree = !(params.num_motion_triangle_steps > 1);


   /*string asset_name(objects[0]->get_asset_name().c_str());
  printf("Object ID: %d", objects[0]->get_object_index());*/

  if (is_embree) {
    foreach (Object *ob, objects) {
        if (ob->get_geometry()->geometry_type == Geometry::HAIR) {
          is_embree = false;
          break;
        }
    }
  }

  /*if (params.top_level) {
    foreach (Object *ob, objects) {
      string asset_name(ob->asset_name.c_str());
      string target_name = "GEO_spring_body.002";
      if (asset_name.compare(target_name) == 0) {
        printf("Object ID: %d", ob->get_object_index());
        break;
      }
    }
}*/
}

#define NODE_TYPE(A) (((A)) & 7u)
#define MAKE_PTR(node_addr, node_type) (((node_type)&7u) | ((node_addr) << 3u))
#define DECODE_NODE(ind) ((ind) >> 3u)
#define PADDING_SIZE 256


struct BVHStack {
  void *node;
  uint child_number;
  uint parent_ptr;

  BVHStack(void *in_node, uint in_child_number, uint in_parent_ptr)
      : node(in_node), child_number(in_child_number), parent_ptr(in_parent_ptr)
  {
  }
};


//void BVHAMD::copy_to_device(Progress & /*progress*/, DeviceScene *dscene)
//{
//
//  if (pack.abvh_nodes.size()) {
//    dscene->bvh_amd.steal_data(pack.abvh_nodes);
//    dscene->bvh_amd.copy_to_device();
//  }
//}


void BVHAMD::pack_instances()
{
    for (size_t i = 0; i < pack.prim_index.size(); i++) {
    if (pack.prim_index[i] != -1) {
      
      pack.prim_index[i] += objects[pack.prim_object[i]]->get_geometry()->prim_offset;
    }
  }

   size_t prim_offset = pack.prim_index.size();


  size_t prim_index_size = pack.prim_index.size();
  size_t prim_tri_verts_size = pack.prim_tri_verts.size();

  size_t pack_prim_index_offset = prim_index_size;
  size_t pack_prim_tri_verts_offset = prim_tri_verts_size;

  size_t amd_bvh_size = pack.abvh_nodes.size();
  size_t amd_bvh_offset = pack.abvh_nodes.size();
  size_t root_offset = amd_bvh_size;


  foreach (Geometry *geom, geometry) {
    BVHAMD *bvh = static_cast<BVHAMD*>(geom->bvh);


    if (geom->need_build_bvh(params.bvh_layout)) {
      prim_index_size += bvh->pack.prim_index.size();
      prim_tri_verts_size += bvh->pack.prim_tri_verts.size();
      amd_bvh_size += bvh->pack.abvh_nodes.size();

    }
  }

  pack.prim_index.resize(prim_index_size);
  pack.prim_type.resize(prim_index_size);
  pack.prim_object.resize(prim_index_size);
  pack.prim_visibility.resize(prim_index_size);
  pack.prim_tri_verts.resize(prim_tri_verts_size);
  pack.prim_tri_index.resize(prim_index_size);

  ASSERT((amd_bvh_size % 256) == 0);
  pack.abvh_nodes.resize(amd_bvh_size);

  uint offset_index = 0;

  if (params.num_motion_curve_steps > 0 || params.num_motion_triangle_steps > 0) {
    pack.prim_time.resize(prim_index_size);
  }

  int *pack_prim_index = (pack.prim_index.size()) ? &pack.prim_index[0] : NULL;
  int *pack_prim_type = (pack.prim_type.size()) ? &pack.prim_type[0] : NULL;
  uint *pack_prim_visibility = (pack.prim_visibility.size()) ? &pack.prim_visibility[0] : NULL;
  float4 *pack_prim_tri_verts = (pack.prim_tri_verts.size()) ? &pack.prim_tri_verts[0] : NULL;
  uint *pack_prim_tri_index = (pack.prim_tri_index.size()) ? &pack.prim_tri_index[0] : NULL;
  float2 *pack_prim_time = (pack.prim_time.size()) ? &pack.prim_time[0] : NULL;
  int *pack_prim_object = (pack.prim_object.size()) ? &pack.prim_object[0] : NULL;

  char *pack_abvhNode = pack.abvh_nodes.size() ? &pack.abvh_nodes[0] : NULL;

  list<Geometry *> geometry_list;

  foreach (Object *ob, objects) {
    Geometry *geom = ob->get_geometry();

    if (!geom->need_build_bvh(params.bvh_layout)) {
      offset_index++;
      continue;
    }

    pack.object_node[offset_index] += root_offset;

    auto it = find(geometry_list.begin(), geometry_list.end(), geom);
    if (it != geometry_list.end()) {
      offset_index++;
      continue;
    }

    geometry_list.insert(it, geom);

    BVHAMD *bvh = static_cast<BVHAMD*>( geom->bvh);

    int geom_prim_offset = geom->prim_offset;



    BoxNodeF32* root = (BoxNodeF32 *)(&bvh->pack.abvh_nodes[0]);
    root->prim_offset = prim_offset;
    offset_index++;


    if (bvh->pack.prim_index.size()) {
      size_t bvh_prim_index_size = bvh->pack.prim_index.size();
      int *bvh_prim_index = &bvh->pack.prim_index[0];
      int *bvh_prim_type = &bvh->pack.prim_type[0];
      uint *bvh_prim_visibility = &bvh->pack.prim_visibility[0];
      uint *bvh_prim_tri_index = &bvh->pack.prim_tri_index[0];
      float2 *bvh_prim_time = bvh->pack.prim_time.size() ? &bvh->pack.prim_time[0] : NULL;

      for (size_t i = 0; i < bvh_prim_index_size; i++) {
        if (bvh->pack.prim_type[i] & PRIMITIVE_ALL_CURVE) {
          pack_prim_index[pack_prim_index_offset] = bvh_prim_index[i] + geom_prim_offset;
          pack_prim_tri_index[pack_prim_index_offset] = -1;
        }
        else {
          pack_prim_index[pack_prim_index_offset] = bvh_prim_index[i] + geom_prim_offset;
          pack_prim_tri_index[pack_prim_index_offset] = bvh_prim_tri_index[i] +
                                                        pack_prim_tri_verts_offset;
        }

        pack_prim_type[pack_prim_index_offset] = bvh_prim_type[i];
        pack_prim_visibility[pack_prim_index_offset] = bvh_prim_visibility[i];
        pack_prim_object[pack_prim_index_offset] = 0;  // unused for instances
        if (bvh_prim_time != NULL) {
          pack_prim_time[pack_prim_index_offset] = bvh_prim_time[i];
        }
        pack_prim_index_offset++;
      }
    }

    /* Merge triangle vertices data. */
    if (bvh->pack.prim_tri_verts.size()) {
      const size_t prim_tri_size = bvh->pack.prim_tri_verts.size();
      memcpy(pack_prim_tri_verts + pack_prim_tri_verts_offset,
             &bvh->pack.prim_tri_verts[0],
             prim_tri_size * sizeof(float4));
      pack_prim_tri_verts_offset += prim_tri_size;
    }

    if (bvh->pack.abvh_nodes.size()) {
      const size_t amd_node_size = bvh->pack.abvh_nodes.size();
      memcpy(pack_abvhNode + amd_bvh_offset, &bvh->pack.abvh_nodes[0], amd_node_size);

    }

    prim_offset += bvh->pack.prim_index.size();
    amd_bvh_offset += bvh->pack.abvh_nodes.size();
  }
  //#endif
}


void BVHAMD::build(Progress &progress, Stats *stats)
{
  progress.set_substatus("Building BVH");


  
  pack.abvh_nodes.clear();
  pack.prim_tri_index.clear();
  pack.prim_tri_verts.clear();
  pack.prim_visibility.clear();
  pack.prim_object.clear();
  pack.prim_type.clear();
  pack.prim_index.clear();
  pack.prim_tri_index.clear();
  pack.prim_time.clear();
  pack.object_node.clear();

  Root_Node root = createEmbreePrimitive(progress);
  /*Root_Node root;
  is_embree = false;

   BVHBuild bvh_build(objects,
                            pack.prim_type,
                            pack.prim_index,
                            pack.prim_object,
                            pack.prim_time,
                            params,
                            progress);

   root.bvh2 = bvh_build.run();*/
  
  pack_primitives();
  


  if (params.top_level) {

    uint offset_index = 0;
    uint amd_bvh_offset = 0;
    pack.object_node.resize(objects.size());
    map<Geometry *, uint> instance_offset;

    foreach (Object *ob, objects) {
      Geometry *geom = ob->get_geometry();

      if (!geom->need_build_bvh(params.bvh_layout)) {
        pack.object_node[offset_index] = 0;
        offset_index++;
        continue;
      }

      map<Geometry *, uint>::iterator it = instance_offset.find(geom);

      if (instance_offset.find(geom) != instance_offset.end()) {

        pack.object_node[offset_index] = it->second;

        offset_index++;

        continue;
      }

      BVHAMD *bvh = static_cast<BVHAMD*> (geom->bvh);

      pack.object_node[offset_index] = amd_bvh_offset;
      instance_offset[geom] = amd_bvh_offset;

      offset_index++;

      amd_bvh_offset += bvh->pack.abvh_nodes.size();
    }
  }


   if (params.top_level) {

    uint offset_index = 0;
     uint amd_bvh_offset = pack.prim_index.size();
    primitives_offset.resize(objects.size());
    map<Geometry *, uint> instance_offset;

    foreach (Object *ob, objects) {
      Geometry *geom = ob->get_geometry();

      if (!geom->need_build_bvh(params.bvh_layout)) {
        offset_index++;
        continue;
      }

      map<Geometry *, uint>::iterator it = instance_offset.find(geom);

      if (instance_offset.find(geom) != instance_offset.end()) {

        primitives_offset[offset_index] = it->second;

        offset_index++;

        continue;
      }

      BVHAMD *bvh = static_cast<BVHAMD*>( geom->bvh);

      primitives_offset[offset_index] = amd_bvh_offset;
      instance_offset[geom] = amd_bvh_offset;

      offset_index++;

      amd_bvh_offset += bvh->pack.prim_index.size();
    }
  }




  if (!is_embree) {

    pack_nodes(root.bvh2);
    root.bvh2->deleteSubtree();
  }
  else {

    if (embree_primitives.size() == 0) {
      ABVHNode current;

      current.tri.shape_id = 0;
      current.tri.prim_type = 0;
      pack.root_index = 1 << 3;

      pack.abvh_nodes.resize(PADDING_SIZE);
      memccpy(&pack.abvh_nodes[0], &current, 1, sizeof(current));
      return;
    }

    if (root.embree->is_leaf()) {

      ABVHNode current;

      current.tri.shape_id = 0;
      current.tri.prim_type = 0;
      pack.root_index = 1 << 3;

      pack.abvh_nodes.resize(PADDING_SIZE);

      memccpy(&pack.abvh_nodes[0], &current, 1, sizeof(current));

      return;
    }

    int tentative_size = (embree_primitives.size() + 100) * 4;

    uint abvh_cnt = 0;

    pack.abvh_nodes.resize(tentative_size * sizeof(ABVHNode));
    
    flatten_nodes_embree(root.embree, abvh_cnt);

    size_t amd_node_size = abvh_cnt * sizeof(ABVHNode);


    size_t hw_alighned = align_up(amd_node_size, PADDING_SIZE);

    pack.abvh_nodes.resize(hw_alighned);

    if (params.top_level) {
      BoxNodeF32 *root = (BoxNodeF32 *)(&pack.abvh_nodes[0]);
      root->object_id = hw_alighned;
    }
  }

  if (params.top_level)
    pack_instances();


}

void BVHAMD::pack_nodes(const BVHNode *root)
{ 

  pack.abvh_nodes.clear();

 if (root->is_leaf()) {

    ABVHNode current;

    current.tri.shape_id = 0;
    current.tri.prim_type = 0;
    pack.root_index = 1 << 3;

    pack.abvh_nodes.resize(PADDING_SIZE);
    memccpy(&pack.abvh_nodes[0], &current, 1, sizeof(current));
    return;
  }

  const size_t num_nodes = root->getSubtreeSize(BVH_STAT_NODE_COUNT);
  const size_t num_leaf_nodes = root->getSubtreeSize(BVH_STAT_LEAF_COUNT);
  assert(num_leaf_nodes <= num_nodes);
  const size_t num_inner_nodes = num_nodes - num_leaf_nodes;

  int tentative_size = (num_leaf_nodes + 100) * 4; //is it a good estimate?

  uint abvh_cnt = 0;



  pack.abvh_nodes.resize(tentative_size*sizeof(ABVHNode));
  flatten_nodes(root, abvh_cnt);

  size_t amd_node_size = abvh_cnt * sizeof(ABVHNode);

  size_t hw_alighned = align_up(amd_node_size, PADDING_SIZE);

  pack.abvh_nodes.resize(hw_alighned);
  if (params.top_level) {
    BoxNodeF32 *top_node = (BoxNodeF32 *)(&pack.abvh_nodes[0]);
    top_node->object_id = hw_alighned;
  }
}

/*void BVHAMD::refit_nodes()
{
  VLOG(1) << "Refit is not implemented ";
}*/

uint BVHAMD::pack_leaf(void *node,
                        uint &abvh_cnt)
{
  uint encoded_ptr = INVALID_NODE;
  uint node_type = 0;
  uint prim_id = 0;
  int object_id = 0;

  if (is_embree) {
    node_type = ((EmbreeNode *)(node))->nodeType;
    object_id = params.top_level ? ((EmbreeCompactInstanceNode *)(node))->object_id : 0;

    prim_id = ((EmbreeCompactTriangleNode *)(node))->prim_id;
  }
  else {
    const LeafNode *leaf = reinterpret_cast<const LeafNode *>(node);
    //obj_id = leaf->hi;
    if (pack.prim_index[leaf->lo] == -1) {
      assert(params.top_level);
      node_type = Object_Node;
      int prim_index_local = (-(~leaf->lo) - 1);
      prim_id = pack.prim_object[prim_index_local];
    }
    else {
      prim_id = leaf->lo;
    }

  }

  ABVHNode abvh_leaf;

  if (node_type == Object_Node) {
    assert(params.top_level);
 
    Transform tfm = transform_quick_inverse(objects[prim_id]->get_tfm());
    abvh_leaf.object.tfm = tfm;
    abvh_leaf.object.shape_id = prim_id;
    /*if (prim_id == 465772)
      int x = 0;*/
    Geometry *geom = objects[prim_id]->get_geometry();
    abvh_leaf.object.unused[0] = pack.object_node[prim_id]; //offset from the root node
    abvh_leaf.object.unused[1] = primitives_offset[prim_id];
    abvh_leaf.object.used = 0;

  }
  else {
    int prim_type = pack.prim_type[prim_id];
    abvh_leaf.tri.triangle_id = 4;
    abvh_leaf.tri.shape_id = prim_id;
    abvh_leaf.tri.prim_type = prim_type;
    abvh_leaf.tri.prim_visibility = pack.prim_visibility[prim_id];

    if ((PRIMITIVE_TRIANGLE & prim_type) || (prim_type & PRIMITIVE_MOTION_TRIANGLE)) {
      uint tri_vindex = pack.prim_tri_index[prim_id];
      float3 vx1_ = float4_to_float3(pack.prim_tri_verts[tri_vindex]);
      float3 vx2_ = float4_to_float3(pack.prim_tri_verts[tri_vindex + 1]);
      float3 vx3_ = float4_to_float3(pack.prim_tri_verts[(tri_vindex + 2)]);

      float3 vx1 = vx1_;
      float3 vx2 = (vx2_);
      float3 vx3 = (vx3_);
      abvh_leaf.tri.unused[0] = pack.prim_object[prim_id];
      if (params.top_level) {
        if (!is_embree)
          object_id = pack.prim_object[prim_id];

        Object *ob = objects[object_id];
        if (!ob->get_geometry()->transform_applied) {
          vx1 = transform_point(&ob->get_tfm(), vx1_);
          vx2 = transform_point(&ob->get_tfm(), vx2_);
          vx3 = transform_point(&ob->get_tfm(), vx3_);
        }
        else {
          abvh_leaf.tri.unused[0] = object_id;
          object_id = OBJECT_NONE;
        }
      }

      TriangleNode tri(vx1, vx2, vx3);
      memcpy(&abvh_leaf.tri.vertices, tri.vertices, sizeof(tri.vertices));

      abvh_leaf.tri.unused[1] = object_id;
      abvh_leaf.tri.unused[2] = 0;

    }
    else {  // hair

      //abvh_leaf.tri.prim_type = prim_type;
      // node_type = Hair_Node;
      abvh_leaf.hair.prim_object = pack.prim_object[prim_id];
      abvh_leaf.hair.prim_idx = pack.prim_index[prim_id] +
                                objects[pack.prim_object[prim_id]]->get_geometry()->prim_offset;

      const int curve_index = abvh_leaf.hair.prim_idx;
      const int segment = prim_type >> PRIMITIVE_NUM_TOTAL;
      Object *ob = objects[pack.prim_object[prim_id]];
      Hair *hair = static_cast<Hair *>(ob->get_geometry());
      Hair::Curve curve = hair->get_curve(pack.prim_index[prim_id]);


      float v00x = __int_as_float(curve.first_key);
      float v00y = __int_as_float(curve.num_keys);

      int k0 = __float_as_int(v00x) + segment;

      int ka = max(k0 - 1, __float_as_int(v00x));

      float v00xx = __int_as_float(curve.first_key + hair->curvekey_offset);
      int k1 = __float_as_int(v00xx) + segment + 1;
      int kb = min(k1 + 1, __float_as_int(v00xx) + __float_as_int(v00y) - 1);
      abvh_leaf.hair.k1 = k1;
      abvh_leaf.hair.kb = kb;

      if (prim_type & PRIMITIVE_ALL_MOTION) {  // have to add the curve offset
        abvh_leaf.hair.curve[0].x = __int_as_float(k0);
        abvh_leaf.hair.curve[0].y = __int_as_float(ka);

        if (params.num_motion_curve_steps > 0 || params.num_motion_triangle_steps > 0) {

          float2 prim_time = pack.prim_time[prim_id];
          abvh_leaf.hair.curve[0].z = prim_time.x;
          abvh_leaf.hair.curve[0].w = prim_time.y;
        }
      }
      else {
        const float3* curve_keys = hair->get_curve_keys().data(); 
        const float *curve_radius = hair->get_curve_radius().data();
        
        float3 ck0 = curve_keys[ka];

        abvh_leaf.hair.curve[0] = make_float4(ck0.x, ck0.y, ck0.z, curve_radius[ka]);

        float3 cka = curve_keys[k0];

        abvh_leaf.hair.curve[1] = make_float4(cka.x, cka.y, cka.z, curve_radius[k0]);
      }
    }
  }
  ABVHNode *node_blob = (ABVHNode *)(&pack.abvh_nodes[0]);
  encoded_ptr = MAKE_PTR(abvh_cnt, node_type);
  node_blob[DECODE_NODE(encoded_ptr)] = abvh_leaf;
  abvh_cnt++;
  return encoded_ptr;
}


uint BVHAMD::flatten_nodes(const BVHNode *node, uint &abvh_cnt)
{

  uint encoded_ptr = INVALID_NODE;

  std::vector<BVHStack> stack;
  stack.reserve(BVHParams::MAX_DEPTH * 2);

  stack.push_back(BVHStack((void *)node, INVALID_NODE, INVALID_NODE));
  ABVHNode *node_blob = (ABVHNode *)(&pack.abvh_nodes[0]);
  //BoxNodeF32 *node_blob = (BoxNodeF32*)((ABVHNode *)(&pack.abvh_nodes[0]));

  while (!stack.empty()) {

    BVHStack entry = stack.back();
    stack.pop_back();
    const BVHNode *current_node = static_cast<const BVHNode *>(entry.node);

    if (current_node->is_leaf()) {

      encoded_ptr = pack_leaf(entry.node, abvh_cnt);

      uint parent_index = entry.parent_ptr;
      uint child_index = entry.child_number;

      ((BoxNodeF32 *)(&node_blob[parent_index]))->child[child_index] = encoded_ptr;
    }
    else {

      BVHNode *children[4] = {};
      BVHNode *child0 = current_node->get_child(0);
      BVHNode *child1 = current_node->get_child(1);

      BoundBox bounds_[4];

      size_t numChildren = 2;
      children[0] = child0;
      bounds_[0] = child0->bounds;
      children[1] = child1;
      bounds_[1] = child1->bounds;

     while (numChildren < 4) {
        ssize_t bestIdx = -1;
        float bestArea = neg_inf;
        for (size_t i = 0; i < numChildren; i++) {
          if (children[i]->is_leaf())
            continue;

          float A = bounds_[i].half_area();
          if (A > bestArea) {
            bestArea = A;
            bestIdx = i;
          }
        }
        if (bestIdx < 0)
          break;

        BVHNode *candidate = children[bestIdx];
        children[bestIdx] = candidate->get_child(0);
        bounds_[bestIdx] = candidate->get_child(0)->bounds;
        children[numChildren] = candidate->get_child(1);
        bounds_[numChildren] = candidate->get_child(1)->bounds;

        numChildren++;
      }

      BoxNodeF32 box_f32;
      box_f32.aligned = 0;

      Unaligned_BoxNode box_f32_unaligned;
      box_f32_unaligned.aligned = 0;

      bool valid_nodes[4] = {false, false, false, false};


        for (int i = 0; i < numChildren; i++) {
        if (children[i] && children[i]->is_unaligned) {
          box_f32.aligned++;
          //box_f32_unaligned.aligned++;
          box_f32_unaligned.aligned = PATH_RAY_NODE_UNALIGNED;
          break;
        }
      }


        for (int i = 0; i < numChildren; i++) {
        if (children[i]) {

          valid_nodes[i] = true;

          if (box_f32.aligned) {
            if (i < 2)
              box_f32_unaligned.aligned_space0[i] = BVHUnaligned::compute_node_transform(
                  children[i]->bounds, children[i]->get_aligned_space());
            else
              box_f32_unaligned.aligned_space1[i - 2] = BVHUnaligned::compute_node_transform(
                  children[i]->bounds, children[i]->get_aligned_space());
          }
          else {
            memcpy(
                &box_f32.children_bound_box[6 * i], &children[i]->bounds.min, 3 * sizeof(float));
            memcpy(&box_f32.children_bound_box[6 * i + 3],
                   &children[i]->bounds.max,
                   3 * sizeof(float));
          }
        }
        else {
          assert(0);
        }
      }

      box_f32.visibility = current_node->visibility;

      box_f32_unaligned.visibility = current_node->visibility;


      encoded_ptr = MAKE_PTR(abvh_cnt, BoxNode32);
      if (box_f32.aligned) {
        abvh_cnt += 4;
        *(Unaligned_BoxNode *)(&node_blob[0] + DECODE_NODE(encoded_ptr)) = box_f32_unaligned;
      }
      else {
        abvh_cnt += 2;
        *(BoxNodeF32 *)(&node_blob[0] + DECODE_NODE(encoded_ptr)) = box_f32;
      }

      uint parent_index = entry.parent_ptr;
      if (parent_index != INVALID_NODE) {

        uint child_index = entry.child_number;

        ((BoxNodeF32 *)(&node_blob[parent_index]))->child[child_index] = encoded_ptr;

      }

      encoded_ptr = DECODE_NODE(encoded_ptr);

      for (int i = 0; i < 4; i++) {
        if (valid_nodes[i])
          stack.push_back(BVHStack(children[i], i, encoded_ptr));
        else {
          if (box_f32.aligned) {
            memset(&box_f32_unaligned.aligned_space1[i - 2], 0, sizeof(Transform));
            //box_f32_unaligned.aligned_space1[i - 2] = transform_identity();
          }
          ((BoxNodeF32 *)(&node_blob[encoded_ptr]))->child[i] = INVALID_NODE;
        }
      }
    }
  }

  return encoded_ptr;
}


/*BVHNode *BVHAMD::widen_children_nodes(const BVHNode *root)
{
  return const_cast<BVHNode *>(root);
}*/

static size_t count_curve_segments(Hair *hair)
{
  size_t num = 0, num_curves = hair->num_curves();

  for (size_t i = 0; i < num_curves; i++)
    num += hair->get_curve(i).num_keys - 1;

  return num;
}

static size_t count_primitives(Geometry *geom)
{
  if (geom->geometry_type == Geometry::MESH) {
    Mesh *mesh = static_cast<Mesh *>(geom);
    return mesh->num_triangles();
  }
  else if (geom->geometry_type == Geometry::HAIR) {
    Hair *hair = static_cast<Hair *>(geom);
    return count_curve_segments(hair);
  }

  return 0;
}

void BVHAMD::add_reference_triangles(
    BoundBox &root1, BoundBox &center1, Mesh *mesh, int i, Object *ob)
{

    /*struct Ref {
    BoundBox bound;
      float2 time;
  };

    array<Ref> ref;*/
  const Attribute *attr_mP = NULL;
  if (mesh->has_motion_blur()) {
    attr_mP = mesh->attributes.find(ATTR_STD_MOTION_VERTEX_POSITION);
  }
  bool need_time = false;
  bool repeated_geometry = false;
  const size_t _num_triangles = mesh->num_triangles();
  //ref.resize(_num_triangles);
  size_t prim_object_size, prim_type_size, prim_index_size, primitive_size = 0;
  std::map<Geometry *, uint>::iterator it;
  if (is_embree) {
    thread_scoped_lock lock(bbox_mutex);
    it = geometry_users.find(mesh);
    if (it == geometry_users.end()) {
      geometry_users[mesh] = pack.prim_index.size();
      prim_object_size = pack.prim_object.size();
      pack.prim_object.resize(prim_object_size + _num_triangles);
      prim_type_size = pack.prim_type.size();
      pack.prim_type.resize(prim_type_size + _num_triangles);
      prim_index_size = pack.prim_index.size();
      pack.prim_index.resize(prim_index_size + _num_triangles);
    }
    else {
      repeated_geometry = true;
      prim_index_size = it->second;
    }

    primitive_size = embree_primitives.size();
    embree_primitives.resize(primitive_size + _num_triangles);
  }


  for (uint j = 0; j < _num_triangles; j++) {
    Mesh::Triangle t = mesh->get_triangle(j);

    float3 *verts_ = mesh->get_verts().data();
    float3 *verts = verts_;
    float3 transformed_vert[3];

    if (!mesh->transform_applied && params.top_level) {
      const Transform *ob_tfm = &ob->get_tfm();
      transformed_vert[0] = transform_point(ob_tfm, verts[t.v[0]]);
      transformed_vert[1] = transform_point(ob_tfm, verts[t.v[1]]);
      transformed_vert[2] = transform_point(ob_tfm, verts[t.v[2]]);

      verts = &transformed_vert[0];
      t.v[0] = 0;
      t.v[1] = 1;
      t.v[2] = 2;

    }
    BoundBox bounds = BoundBox::empty;
    if (attr_mP == NULL) {

      t.bounds_grow(verts, bounds);
    }
    else if (params.num_motion_triangle_steps == 0 || params.use_spatial_split) {

      const size_t num_verts = mesh->get_verts().size();
      const size_t num_steps = mesh->get_motion_steps();
      const float3 *vert_steps = attr_mP->data_float3();

      t.bounds_grow(verts, bounds);
      for (size_t step = 0; step < num_steps - 1; step++) {
        if (!mesh->transform_applied && params.top_level) {

          vert_steps += (step * num_verts);

          const Transform *ob_tfm = &ob->get_tfm();
          transformed_vert[0] = transform_point(ob_tfm, vert_steps[t.v[0]]);
          transformed_vert[1] = transform_point(ob_tfm, vert_steps[t.v[1]]);
          transformed_vert[2] = transform_point(ob_tfm, vert_steps[t.v[2]]);

          vert_steps = &transformed_vert[0];
          t.v[0] = 0;
          t.v[1] = 1;
          t.v[2] = 2;

          t.bounds_grow(vert_steps + step * num_verts, bounds);
        }
        else
          t.bounds_grow(vert_steps + step * num_verts, bounds);
      }
    }
    else {
      need_time = true;
      verts = verts_;

      const int num_bvh_steps = params.num_motion_curve_steps * 2 + 1;
      const float num_bvh_steps_inv_1 = 1.0f / (num_bvh_steps - 1);
      const size_t num_verts = mesh->get_verts().size();
      const size_t num_steps = mesh->get_motion_steps();
      const float3 *vert_steps_ = attr_mP->data_float3();
      const float3 *vert_steps = vert_steps_;
      /*float3 vert_steps_transformed[3];
      if (params.top_level && !mesh->transform_applied) {
        const Transform *ob_tfm = &ob->tfm;
        transformed_vert[0] = transform_point(ob_tfm, verts[t.v[0]]);
        transformed_vert[1] = transform_point(ob_tfm, verts[t.v[1]]);
        transformed_vert[2] = transform_point(ob_tfm, verts[t.v[2]]);

        verts = &transformed_vert[0];
        t.v[0] = 3;
        t.v[1] = 1;
        t.v[2] = 2;
      }*/

      float3 prev_verts[3];
      t.motion_verts(verts, vert_steps, num_verts, num_steps, 0.0f, prev_verts);
      if (params.top_level && !mesh->transform_applied) {
        const Transform *ob_tfm = &ob->get_tfm();
        prev_verts[0] = transform_point(ob_tfm, prev_verts[0]);
        prev_verts[1] = transform_point(ob_tfm, prev_verts[1]);
        prev_verts[2] = transform_point(ob_tfm, prev_verts[2]);
      }

      BoundBox prev_bounds = BoundBox::empty;
      prev_bounds.grow(prev_verts[0]);
      prev_bounds.grow(prev_verts[1]);
      prev_bounds.grow(prev_verts[2]);
      /* Create all primitive time steps, */
      for (int bvh_step = 1; bvh_step < num_bvh_steps; ++bvh_step) {
        const float curr_time = (float)(bvh_step)*num_bvh_steps_inv_1;
        float3 curr_verts[3];
        t.motion_verts(verts, vert_steps, num_verts, num_steps, curr_time, curr_verts);
        if (params.top_level && !mesh->transform_applied) {
          const Transform *ob_tfm = &ob->get_tfm();
          curr_verts[0] = transform_point(ob_tfm, curr_verts[0]);
          curr_verts[1] = transform_point(ob_tfm, curr_verts[1]);
          curr_verts[2] = transform_point(ob_tfm, curr_verts[2]);
        }
        BoundBox curr_bounds = BoundBox::empty;
        curr_bounds.grow(curr_verts[0]);
        curr_bounds.grow(curr_verts[1]);
        curr_bounds.grow(curr_verts[2]);
        BoundBox bounds = prev_bounds;
        bounds.grow(curr_bounds);
        if (bounds.valid()) {// Bug: transformation is not applied
          thread_scoped_lock lock(bbox_mutex);
          const float prev_time = (float)(bvh_step - 1) * num_bvh_steps_inv_1;

          if (is_embree) {

            RTCBuildPrimitive prim;

            prim.geomID = params.top_level ? i << 1 : ob->visibility_for_tracing();

            prim.primID = j + prim_index_size;
            prim.lower_x = bounds.min.x;
            prim.lower_y = bounds.min.y;
            prim.lower_z = bounds.min.z;
            prim.upper_x = bounds.max.x;
            prim.upper_y = bounds.max.y;
            prim.upper_z = bounds.max.z;
            embree_primitives.push_back(prim);

          }
          else {

            reference.push_back(
                BVHReference(bounds, j, i, PRIMITIVE_MOTION_TRIANGLE, prev_time, curr_time));
            root_bounds.grow(bounds);
            center.grow(bounds.center2());
          }
        }
        prev_bounds = curr_bounds;
      }
    }

    if (bounds.valid()) {

         int prim_type = (attr_mP == NULL) ? PRIMITIVE_TRIANGLE : PRIMITIVE_MOTION_TRIANGLE;

         thread_scoped_lock lock(bbox_mutex);

        if (is_embree && !repeated_geometry) {
           // not a one to one mapping, more than one object can share the same geometry, this
           // array cannot used when multiple instances generate multiple bvhs
            // and there is no one to one mapping between prim_id and object_id
            // there is only one to one mapping between embree_primitives and object_id
          pack.prim_object[prim_object_size + j] = i; 
          pack.prim_type[prim_type_size + j] = prim_type;
          pack.prim_index[prim_index_size + j] = j;
        }


        if (is_embree) {

          RTCBuildPrimitive prim;

          prim.geomID = params.top_level ? i << 1 : ob->visibility_for_tracing();

          prim.primID = j + prim_index_size;
          prim.lower_x = bounds.min.x;
          prim.lower_y = bounds.min.y;
          prim.lower_z = bounds.min.z;
          prim.upper_x = bounds.max.x;
          prim.upper_y = bounds.max.y;
          prim.upper_z = bounds.max.z;
          embree_primitives[j + primitive_size] = prim;
          //embree_primitives.push_back(prim);
        }
        else if (!need_time) {
          reference.push_back(BVHReference(bounds, j, i, prim_type));
          root_bounds.grow(bounds);
          center.grow(bounds.center2());
        }
    }
  }

}

void BVHAMD::add_reference_curves(BoundBox &root1, BoundBox &center1, Hair *hair, int i, Object *ob)
{
  const Attribute *curve_attr_mP = NULL;
  if (hair->has_motion_blur()) {
    curve_attr_mP = hair->attributes.find(ATTR_STD_MOTION_VERTEX_POSITION);
  }
  const PrimitiveType primitive_type =
      (curve_attr_mP != NULL) ?
          ((hair->curve_shape == CURVE_RIBBON) ? PRIMITIVE_MOTION_CURVE_RIBBON :
                                                 PRIMITIVE_MOTION_CURVE_THICK) :
          ((hair->curve_shape == CURVE_RIBBON) ? PRIMITIVE_CURVE_RIBBON : PRIMITIVE_CURVE_THICK);


#ifdef SINGLE_LEVEL
  float3 c0 = transform_get_column(&ob->tfm, 0);
  float3 c1 = transform_get_column(&ob->tfm, 1);
  float3 c2 = transform_get_column(&ob->tfm, 2);
  float scalar = powf(fabsf(dot(cross(c0, c1), c2)), 1.0f / 3.0f);
#endif
  const size_t num_curves = hair->num_curves();
  for (uint j = 0; j < num_curves; j++) {
    const Hair::Curve curve = hair->get_curve(j);
    const float *curve_radius = hair->get_curve_radius().data();
    for (int k = 0; k < curve.num_keys - 1; k++) {
      if (curve_attr_mP == NULL) {
        BoundBox bounds = BoundBox::empty;

#ifdef SINGLE_LEVEL
        if (!hair->transform_applied) {
          float3 P[4];

          P[0] = transform_point(&ob->tfm, hair->curve_keys[max(first_key + k - 1, first_key)]);
          P[1] = transform_point(&ob->tfm, hair->curve_keys[first_key + k]);
          P[2] = transform_point(&ob->tfm, hair->curve_keys[first_key + k + 1]);
          P[3] = transform_point(
              &ob->tfm,
              hair->curve_keys[min(first_key + k + 2, first_key + hair->num_keys() - 1)]);

          float radius1 = curve_radius[first_key + k] * scalar;
          float radius2 = curve_radius[first_key + k + 1] * scalar;
          float mr = max(curve_radius[first_key + k], curve_radius[first_key + k + 1]);
          mr *= scalar;

          float3 lower;
          float3 upper;

          bounds.grow(lower, mr);
          bounds.grow(upper, mr);

          curvebounds(&lower.x, &upper.x, P, 0);
          curvebounds(&lower.y, &upper.y, P, 1);
          curvebounds(&lower.z, &upper.z, P, 2);
        }
        else
          curve.bounds_grow(k, &hair->curve_keys[0], curve_radius, bounds);
        #else
        curve.bounds_grow(k, hair->get_curve_keys().data(), curve_radius, bounds);
#endif


        if (bounds.valid()) {
          int packed_type = PRIMITIVE_PACK_SEGMENT(primitive_type, k);

          thread_scoped_lock lock(bbox_mutex);
          reference.push_back(BVHReference(bounds, j, i, packed_type));
          root_bounds.grow(bounds);
          center.grow(bounds.center2());  

        }
      }
      else if (params.num_motion_curve_steps == 0 || params.use_spatial_split) {
       
        BoundBox bounds = BoundBox::empty;
        curve.bounds_grow(k, hair->get_curve_keys().data(), curve_radius, bounds);
        const size_t num_keys = hair->get_curve_keys().size();
        const size_t num_steps = hair->get_motion_steps();
        const float3 *key_steps = curve_attr_mP->data_float3();
        for (size_t step = 0; step < num_steps - 1; step++) {
          curve.bounds_grow(k, key_steps + step * num_keys, curve_radius, bounds);
        }
        if (bounds.valid()) {

          thread_scoped_lock lock(bbox_mutex);
          int packed_type = PRIMITIVE_PACK_SEGMENT(primitive_type, k);
          reference.push_back(BVHReference(bounds, j, i, packed_type));
          root_bounds.grow(bounds);
          center.grow(bounds.center2());
        }
      }
      else {
        /* Motion curves, trace optimized case:  we split curve keys
         * primitives into separate nodes for each of the time steps.
         * This way we minimize overlap of neighbor curve primitives.
         */
        const int num_bvh_steps = params.num_motion_curve_steps * 2 + 1;
        const float num_bvh_steps_inv_1 = 1.0f / (num_bvh_steps - 1);
        const size_t num_steps = hair->get_motion_steps();
        const float3 *curve_keys = hair->get_curve_keys().data();
        const float3 *key_steps = curve_attr_mP->data_float3();
        const size_t num_keys = hair->get_curve_keys().size();
        /* Calculate bounding box of the previous time step.
         * Will be reused later to avoid duplicated work on
         * calculating BVH time step boundbox.
         */
        float4 prev_keys[4];
        curve.cardinal_motion_keys(curve_keys,
                                   curve_radius,
                                   key_steps,
                                   num_keys,
                                   num_steps,
                                   0.0f,
                                   k - 1,
                                   k,
                                   k + 1,
                                   k + 2,
                                   prev_keys);
        BoundBox prev_bounds = BoundBox::empty;
        curve.bounds_grow(prev_keys, prev_bounds);
        /* Create all primitive time steps, */
        for (int bvh_step = 1; bvh_step < num_bvh_steps; ++bvh_step) {
          const float curr_time = (float)(bvh_step)*num_bvh_steps_inv_1;
          float4 curr_keys[4];
          curve.cardinal_motion_keys(curve_keys,
                                     curve_radius,
                                     key_steps,
                                     num_keys,
                                     num_steps,
                                     curr_time,
                                     k - 1,
                                     k,
                                     k + 1,
                                     k + 2,
                                     curr_keys);
          BoundBox curr_bounds = BoundBox::empty;
          curve.bounds_grow(curr_keys, curr_bounds);
          BoundBox bounds = prev_bounds;
          bounds.grow(curr_bounds);
          if (bounds.valid()) {
            const float prev_time = (float)(bvh_step - 1) * num_bvh_steps_inv_1;
            int packed_type = PRIMITIVE_PACK_SEGMENT(primitive_type, k);
            thread_scoped_lock lock(bbox_mutex);
            reference.push_back(BVHReference(bounds, j, i, packed_type, prev_time, curr_time));
            root_bounds.grow(bounds);
            center.grow(bounds.center2());
          }
          /* Current time boundbox becomes previous one for the
           * next time step.
           */
          prev_bounds = curr_bounds;
        }
      }
    }
  }

  //thread_scoped_lock lock(bbox_mutex);
 #ifdef SINGLE_LEVEL
  /*if(is_embree)*/ {
    std::map<Hair *, uint>::iterator it;
    //
    it = hair_list.find(hair);

    if (it != hair_list.end()) {
      size_t rtc_index = 0;
        for (size_t j = 0; j < num_curves; ++j) {
        Hair::Curve c = hair->get_curve(j);
        for (size_t k = 0; k < c.num_segments(); ++k) {
          
          reference.push_back(BVHReference(bounds_list[rtc_index],
                              j + it->second,
                              i,
                              (PRIMITIVE_PACK_SEGMENT(primitive_type, k))));

          ++rtc_index;
        }
      }
      return;
  }
    else
      hair_list[hair] = pack.prim_index.size();
  }
#endif

}

void BVHAMD::add_reference_geometry(
    BoundBox &root, BoundBox &center, Geometry *geom, int i, Object *ob)
{
  if (geom->geometry_type == Geometry::MESH) {
    Mesh *mesh = static_cast<Mesh *>(geom);
    add_reference_triangles(root, center, mesh, i, ob);
  }
  else if (geom->geometry_type == Geometry::HAIR) {
    Hair *hair = static_cast<Hair *>(geom);
    add_reference_curves(root, center, hair, i, ob);
  }
}

void BVHAMD::add_reference_object(BoundBox &root1, BoundBox &center1, Object *ob, int i)
{
  //thread_scoped_lock lock(bbox_mutex);
  if (is_embree) {
    RTCBuildPrimitive prim;
    prim.geomID = 1 | (ob->visibility_for_tracing() << 1);
    prim.primID = i;
    prim.lower_x = ob->bounds.min.x;
    prim.lower_y = ob->bounds.min.y;
    prim.lower_z = ob->bounds.min.z;
    prim.upper_x = ob->bounds.max.x;
    prim.upper_y = ob->bounds.max.y;
    prim.upper_z = ob->bounds.max.z;
    thread_scoped_lock lock(bbox_mutex);
    embree_primitives.push_back(prim);

    pack.prim_index.push_back_slow(-1);
    pack.prim_object.push_back_slow(i);
    pack.prim_type.push_back_slow(PRIMITIVE_NONE);
  }
  else {
    thread_scoped_lock lock(bbox_mutex);
    reference.push_back(BVHReference(ob->bounds, -1, i, 0));
    root_bounds.grow(ob->bounds);
    center.grow(ob->bounds.center2());
  }

}

void BVHAMD::add_reference(Object *ob, int i)
{
  BoundBox root_;
  Geometry *geom = ob->get_geometry();
  if (geom->is_instanced())
    add_reference_object(root_, root_, ob, i);
  else
    add_reference_geometry(root_, root_, geom, i, ob);
}
/*
static void splitPrimitive(const struct RTCBuildPrimitive *primitive,
                           unsigned int dimension,
                           float position,
                           struct RTCBounds *leftBounds,
                           struct RTCBounds *rightBounds,
                           void *userPtr)
{
  assert(dimension < 3);
  std::array<float3, 3> newVertex;

  Mesh::Triangle tri = ((Mesh *)userPtr)->get_triangle(primitive->primID);
  
  // create split points on the edges.
  int count = 0;
  for (int i = 0; i < 3; i++) {
    float3 v0 = ((Mesh *)userPtr)->verts[i];
    float3 v1 = ((Mesh *)userPtr)->verts[(i + 1) % 3];
    float tmpFactor = (position - v0[dimension]) / (v1[dimension] - v0[dimension]);
    if (tmpFactor > 0 && tmpFactor < 1) {
      newVertex[count++] = v0 + (v1 - v0) * tmpFactor;
    }
  }
  constexpr float floatMin = std::numeric_limits<float>::lowest();
  constexpr float floatMax = std::numeric_limits<float>::max();

  // AABB of everything < position
  {
    BoundBox aabb = BoundBox::empty;
    for (int i = 0; i < 3; i++) {
      float currentPos;
      if (dimension == 0)
        currentPos = ((Mesh *)userPtr)->verts[i].x;
      if (dimension == 1)
        currentPos = ((Mesh *)userPtr)->verts[i].y;
      if (dimension == 2)
        currentPos = ((Mesh *)userPtr)->verts[i].z;
      if (currentPos <= position) {
        aabb.grow(((Mesh *)userPtr)->verts[i]);
      }
    }

    for (int i = 0; i < count; i++) {
      aabb.grow(newVertex[i]);
    }

    leftBounds->lower_x = aabb.min.x;
    leftBounds->lower_y = aabb.min.y;
    leftBounds->lower_z = aabb.min.z;

    leftBounds->upper_x = aabb.max.x;
    leftBounds->upper_y = aabb.max.y;
    leftBounds->upper_z = aabb.max.z;
  }
  // AABB of everything > position
  {
    BoundBox aabb = BoundBox::empty;
    for (int i = 0; i < 3; i++) {
      float currentPos;
      if (dimension == 0)
        currentPos = ((Mesh *)userPtr)->verts[i].x;
      if (dimension == 1)
        currentPos = ((Mesh *)userPtr)->verts[i].y;
      if (dimension == 2)
        currentPos = ((Mesh *)userPtr)->verts[i].z;
      if (currentPos <= position) {
        aabb.grow(((Mesh *)userPtr)->verts[i]);
      }
    }

    for (int i = 0; i < count; i++) {
      aabb.grow(newVertex[i]);
    }

    rightBounds->lower_x = aabb.min.x;
    rightBounds->lower_y = aabb.min.y;
    rightBounds->lower_z = aabb.min.z;

    rightBounds->upper_x = aabb.max.x;
    rightBounds->upper_y = aabb.max.y;
    rightBounds->upper_z = aabb.max.z;
  }
}*/
  

void BVHAMD::pack_triangle(int idx, float4 tri_verts[3])
{
  int tob = pack.prim_object[idx];
  assert(tob >= 0 && tob < objects.size());
  const Mesh *mesh = static_cast<const Mesh *>(objects[tob]->get_geometry());

  int tidx = pack.prim_index[idx];
  Mesh::Triangle t = mesh->get_triangle(tidx);
  const float3 *vpos = mesh->get_verts().data();
  float3 v0 = vpos[t.v[0]];
  float3 v1 = vpos[t.v[1]];
  float3 v2 = vpos[t.v[2]];

  tri_verts[0] = float3_to_float4(v0);
  tri_verts[1] = float3_to_float4(v1);
  tri_verts[2] = float3_to_float4(v2);
}


void BVHAMD::pack_primitives()
{
  const size_t tidx_size = pack.prim_index.size();
  size_t num_prim_triangles = 0;
  /* Count number of triangles primitives in BVH. */
  for (unsigned int i = 0; i < tidx_size; i++) {
    if ((pack.prim_index[i] != -1)) {
      if ((pack.prim_type[i] & PRIMITIVE_ALL_TRIANGLE) != 0) {
        ++num_prim_triangles;
      }
    }
  }
  /* Reserve size for arrays. */
  pack.prim_tri_index.clear();
  pack.prim_tri_index.resize(tidx_size);
  pack.prim_tri_verts.clear();
  pack.prim_tri_verts.resize(num_prim_triangles * 3);
  pack.prim_visibility.clear();
  pack.prim_visibility.resize(tidx_size);
  /* Fill in all the arrays. */
  size_t prim_triangle_index = 0;
  for (unsigned int i = 0; i < tidx_size; i++) {
    if (pack.prim_index[i] != -1) {
      int tob = pack.prim_object[i];
      Object *ob = objects[tob];
      if ((pack.prim_type[i] & PRIMITIVE_ALL_TRIANGLE) != 0) {
        pack_triangle(i, (float4 *)&pack.prim_tri_verts[3 * prim_triangle_index]);
        pack.prim_tri_index[i] = 3 * prim_triangle_index;
        ++prim_triangle_index;
      }
      else {
        pack.prim_tri_index[i] = -1;
      }
      pack.prim_visibility[i] = ob->visibility_for_tracing();
    }
    else {
      pack.prim_tri_index[i] = -1;
      pack.prim_visibility[i] = 0;
    }
  }
}


Root_Node BVHAMD::createEmbreePrimitive(Progress &progress)
{
  size_t num_alloc_references = 0;
  Root_Node bvh_root;
  bvh_root.embree = NULL;
  bvh_root.bvh2 = NULL;


  foreach (Object *ob, objects) {
    if (params.top_level) {
      if (!ob->is_traceable()) {
        continue;
      }
      if (!ob->get_geometry()->is_instanced()) {
        num_alloc_references += count_primitives(ob->get_geometry());
      }
      else
        num_alloc_references++;
    }
    else {
      num_alloc_references += count_primitives(ob->get_geometry());
    }
  }

  if (num_alloc_references == 0) {
    return bvh_root;
  }

  reference.reserve(num_alloc_references);
  embree_primitives.reserve(num_alloc_references);

  pack.prim_object.reserve(num_alloc_references);
  pack.prim_index.reserve(num_alloc_references);

  int i = 0;

  TaskPool pool;

  foreach (Object *ob, objects) {
    if (params.top_level) {
      if (!ob->is_traceable()) {
        ++i;
        continue;
      }
      //pool.push([=] { add_reference(ob, i); });
      add_reference(ob, i);

    }
    else
      add_reference_geometry(root_bounds, center, ob->get_geometry(), i, ob);

    i++;
  }

  pool.wait_work();

  if(!is_embree) {

      BVHBuild_Custom bvh_build(objects,
                       pack.prim_type,
                       pack.prim_index,
                       pack.prim_object,
                       pack.prim_time,
                       params,
                       progress);

      bvh_build.set_reference(reference, root_bounds, center);
      bvh_root.bvh2 = bvh_build.run();

  }
  else {

  RTCDevice rtc_device = rtcNewDevice("verbose=0");

  RTCBVH bvh = rtcNewBVH(rtc_device);

  RTCBuildArguments arguments = rtcDefaultBuildArguments();

  bool leafSplit = false;  // true;
  bool collapse = false;  // true;

  arguments.byteSize = sizeof(arguments);
  arguments.buildFlags = RTC_BUILD_FLAG_DYNAMIC;
  arguments.primitives = embree_primitives.data();
  arguments.bvh = bvh;
  arguments.userPtr = NULL;  // objects.data();
      //geometry.data();
  arguments.primitiveCount = embree_primitives.size();
  arguments.primitiveArrayCapacity = embree_primitives.capacity();

  if (params.top_level) {
    arguments.buildQuality = RTC_BUILD_QUALITY_MEDIUM;
    arguments.userPtr = objects.data();
    arguments.maxBranchingFactor = 4;
    arguments.maxDepth = 1024;
    arguments.sahBlockSize = 1;
    arguments.minLeafSize = 1;
    arguments.maxLeafSize = 1;
    arguments.traversalCost = 1.0f;
    arguments.intersectionCost = 1.0f;
    arguments.createNode = EmbreeCompactNode::create;
    arguments.setNodeChildren = EmbreeCompactNode::setChildren;
    arguments.setNodeBounds = EmbreeCompactNode::setBounds;
    arguments.createLeaf = EmbreeCompactInstanceNode::create;  
  }
  else {

    if (leafSplit) {
      arguments.buildQuality = RTC_BUILD_QUALITY_HIGH;
    }
    else {
      arguments.buildQuality = RTC_BUILD_QUALITY_MEDIUM;
    }
    if (collapse) {
      arguments.maxBranchingFactor = 2;
    }
    else
      arguments.maxBranchingFactor = 4;

    arguments.maxDepth = 1024;
    arguments.sahBlockSize = 1;
    arguments.minLeafSize = 1;
    arguments.maxLeafSize = 1;
    arguments.traversalCost = 1.0f;
    arguments.intersectionCost = 1.0f;
    
    arguments.createNode = EmbreeCompactNode::create;
    arguments.setNodeChildren = EmbreeCompactNode::setChildren;
    arguments.setNodeBounds = EmbreeCompactNode::setBounds;
    arguments.createLeaf = EmbreeCompactTriangleNode::create;
    arguments.splitPrimitive = NULL;  // splitPrimitive;
    
  }

  bvh_root.embree  = (EmbreeNode *)rtcBuildBVH(&arguments);

  //rtcReleaseBVH(bvh);
  }

  //return root;
  return bvh_root;
}

uint BVHAMD::pack_node_embree(EmbreeCompactNode *node,
                              uint parent_index,
                              uint &abvh_cnt,
                              bool **valid_nodes,
                              EmbreeCompactNode** children
    )
{


  EmbreeCompactNode *node_ = (EmbreeCompactNode *)node;
  BoxNodeF32 compactNode;
   for (uint32_t i = 0; i < 4; i++) {
     if (node_->pointer[i] != nullptr) {
       children[i] = (EmbreeCompactNode *)node->pointer[i];
       *valid_nodes[i] = true;
       memcpy(&compactNode.children_bound_box[6 * i],
              &node_->boundingBoxes[i].min,
              3 * sizeof(float));
       memcpy(&compactNode.children_bound_box[6 * i + 3],
              &node_->boundingBoxes[i].max,
              3 * sizeof(float));
     }
     else {
       compactNode.child[i] = INVALID_NODE;
     }
   }
   compactNode.aligned = 0;
   compactNode.visibility = node->visibility;
   uint node_id_ptr = MAKE_PTR(abvh_cnt, BoxNode32);
   abvh_cnt += 2;
   ABVHNode *abvh_node = (ABVHNode *)(&pack.abvh_nodes[0]);

   *(BoxNodeF32 *)(&abvh_node[0] + DECODE_NODE(node_id_ptr)) = compactNode;
   return node_id_ptr;
}

void BVHAMD::flatten_nodes_embree(EmbreeNode *node, uint &abvh_cnt)
{
  uint encoded_ptr = INVALID_NODE;
  ABVHNode *node_blob = (ABVHNode *)(&pack.abvh_nodes[0]);

  std::vector<BVHStack> stack;

  stack.reserve(BVHParams::MAX_DEPTH * 2);

  stack.push_back(BVHStack(node, INVALID_NODE, INVALID_NODE));

  while (!stack.empty()) {

    BVHStack entry = stack.back();
    stack.pop_back();
    if (((EmbreeNode*)(entry.node))->is_leaf()) {
      uint parent_index = entry.parent_ptr;
      uint child_index = entry.child_number;
      encoded_ptr = pack_leaf(entry.node, abvh_cnt);

      ((BoxNodeF32 *)(&node_blob[parent_index]))->child[child_index] = encoded_ptr;
    }
    else {

      bool valid[4]= {false, false, false, false};
      bool *valid_nodes[4] = {&valid[0], &valid[1], &valid[2], &valid[3]};

      EmbreeCompactNode *children[4];
      uint parent_index = entry.parent_ptr;
      encoded_ptr = pack_node_embree(
          (EmbreeCompactNode *)entry.node, parent_index, abvh_cnt, valid_nodes, children);


      if (parent_index != INVALID_NODE) {

        uint child_index = entry.child_number;
        ((BoxNodeF32 *)(&node_blob[parent_index]))->child[child_index] = encoded_ptr;
        //((BoxNodeF32 *)(&node_blob[parent_index]))->visibility = node->visibility;

      }

      encoded_ptr = DECODE_NODE(encoded_ptr);

      for (int i = 0; i < 4; i++) {
        if (valid[i])
          stack.push_back(BVHStack(children[i], i, encoded_ptr));
        else
          ((BoxNodeF32*)(&node_blob[encoded_ptr]))->child[i] = INVALID_NODE;

      }
    }
  }
}

CCL_NAMESPACE_END
