//#include "bvh/bvh_amd.h"
#include "render/mesh.h"
#include "render/object.h"
#include "util/util_foreach.h"
#include "util/util_half_float.h"
#include "util/util_logging.h"
#include "util/util_progress.h"

#include "bvh/bvh_node.h"
#include "bvh_amd.h"
#include "render/hair.h"

CCL_NAMESPACE_BEGIN

BVHAMD::BVHAMD(const BVHParams &params,
               const vector<Geometry *> &geometry,
               const vector<Object *> &objects)
    : BVH(params, geometry, objects)
{
}

#define INVALID_NODE 0xffffffffU
#define MAKE_PTR(node_addr, node_type) (((node_type)&7u) | ((node_addr) << 3u))
#define NODE_TYPE(A) (((A)) & 7u)
#define DECODE_NODE(ind) ((ind) >> 3u)
#define PADDING_SIZE 256


void BVHAMD::copy_to_device(Progress & /*progress*/, DeviceScene *dscene)
{

  if (pack.abvh_nodes.size()) {
    dscene->bvh_amd.steal_data(pack.abvh_nodes);
    dscene->bvh_amd.copy_to_device();
  }

  if (pack.offset.size()) {
    dscene->bvh_amd_offset.steal_data(pack.offset);
    dscene->bvh_amd_offset.copy_to_device();
  }
}



void BVHAMD::pack_instances(size_t nodes_size, size_t leaf_nodes_size)
{
  for (size_t i = 0; i < pack.prim_index.size(); i++) {
    if (pack.prim_index[i] != -1) {
      pack.prim_index[i] += objects[pack.prim_object[i]]->geometry->prim_offset;
    }
  }

  size_t prim_offset = pack.prim_index.size();

  pack.object_node.clear();
  pack.offset.clear();

  size_t prim_index_size = pack.prim_index.size();
  size_t prim_tri_verts_size = pack.prim_tri_verts.size();

  size_t pack_prim_index_offset = prim_index_size;
  size_t pack_prim_tri_verts_offset = prim_tri_verts_size;

  size_t amd_bvh_size = pack.abvh_nodes.size();
  size_t amd_bvh_offset = pack.abvh_nodes.size();

  foreach (Geometry *geom, geometry) {
    BVH *bvh = geom->bvh;

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
  pack.offset.resize(objects.size());

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

  map<Geometry *, uint2> geometry_map;

  foreach (Object *ob, objects) {
    Geometry *geom = ob->geometry;

    if (!geom->need_build_bvh(params.bvh_layout)) {
      offset_index++;
      continue;
    }

    map<Geometry *, uint2>::iterator it = geometry_map.find(geom);

    if (geometry_map.find(geom) != geometry_map.end()) {

      uint2 bvh_offset = make_uint2(it->second.x, it->second.y);
      pack.offset[offset_index].x = bvh_offset.x;
      pack.offset[offset_index++].y = bvh_offset.y;
      continue;
    }

    BVH *bvh = geom->bvh;

    int geom_prim_offset = geom->prim_offset;

    pack.offset[offset_index].x = amd_bvh_offset;
    pack.offset[offset_index++].y = prim_offset;

    uint2 offsets = make_uint2(pack.offset[offset_index - 1].x, pack.offset[offset_index - 1].y);
    geometry_map[geom] = offsets;

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
}

void BVHAMD::pack_nodes(const BVHNode *root)
{
  const size_t num_nodes = root->getSubtreeSize(BVH_STAT_NODE_COUNT);
  const size_t num_leaf_nodes = root->getSubtreeSize(BVH_STAT_LEAF_COUNT);
  assert(num_leaf_nodes <= num_nodes);
  const size_t num_inner_nodes = num_nodes - num_leaf_nodes;

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

  int tentative_size = (num_leaf_nodes + 100) * 4;

  array<ABVHNode> abvh_nodes(tentative_size);
  uint abvh_cnt = flatten_nodes(root, abvh_nodes);

  size_t amd_node_size = abvh_cnt * sizeof(ABVHNode);

  size_t hw_alighned = align_up(amd_node_size, PADDING_SIZE);

  pack.abvh_nodes.resize(hw_alighned);

  memcpy(&pack.abvh_nodes[0], &abvh_nodes[0], amd_node_size);

  if (params.top_level) {
    pack_instances(num_inner_nodes, num_leaf_nodes);
  }
}

void BVHAMD::refit_nodes()
{
  VLOG(1) << "Refit is not implemented ";
}

uint BVHAMD::pack_leaf(const BVHNode *node, array<ABVHNode> &abvh_node, uint &abvh_cnt)
{
  uint encoded_ptr = INVALID_NODE;

  ABVHNode abvh_leaf;
  const LeafNode *leaf = reinterpret_cast<const LeafNode *>(node);
  if (params.top_level && pack.prim_index[leaf->lo] == -1) {
    int prim_index = (-(~leaf->lo) - 1);
    abvh_leaf.tri.shape_id = pack.prim_object[prim_index];
    encoded_ptr = MAKE_PTR(abvh_cnt, Object_Node);
    Transform tfm = transform_quick_inverse(objects[abvh_leaf.tri.shape_id]->tfm);
    abvh_leaf.object.tfm = tfm;
  }
  else {
    int prim_type = pack.prim_type[leaf->lo];
    int prim_id = leaf->lo;
    abvh_leaf.tri.triangle_id = 4;
    abvh_leaf.tri.shape_id = prim_id;
    abvh_leaf.tri.prim_type = prim_type;
    abvh_leaf.tri.prim_visibility = pack.prim_visibility[leaf->lo];

    if ((PRIMITIVE_TRIANGLE & prim_type) || (prim_type & PRIMITIVE_MOTION_TRIANGLE)) {
      uint tri_vindex = pack.prim_tri_index[prim_id];
      float4 vx1 = pack.prim_tri_verts[tri_vindex];
      float4 vx2 = pack.prim_tri_verts[tri_vindex + 1];
      float4 vx3 = pack.prim_tri_verts[(tri_vindex + 2)];
      float4 vx4 = make_float4(0.f, 0.f, 0.f, 0.f);

      abvh_leaf.tri.v0x = vx1.x;
      abvh_leaf.tri.v0y = vx1.y;
      abvh_leaf.tri.v0z = vx1.z;

      abvh_leaf.tri.v1x = vx2.x;
      abvh_leaf.tri.v1y = vx2.y;
      abvh_leaf.tri.v1z = vx2.z;

      abvh_leaf.tri.v2x = vx3.x;
      abvh_leaf.tri.v2y = vx3.y;
      abvh_leaf.tri.v2z = vx3.z;

      abvh_leaf.tri.data0 = __int_as_float(pack.prim_object[prim_id]);
      abvh_leaf.tri.data1 = vx4.y;
      abvh_leaf.tri.data2 = vx4.z;
    }
    else {  // hair

      abvh_leaf.hair.prim_object = pack.prim_object[prim_id];
      abvh_leaf.hair.prim_idx = pack.prim_index[prim_id] +
                                objects[pack.prim_object[prim_id]]->geometry->prim_offset;

      Object *ob = objects[pack.prim_object[prim_id]];

      Hair *hair = static_cast<Hair *>(ob->geometry);
      Hair::Curve curve = hair->get_curve(pack.prim_index[prim_id]);

      float v00x = __int_as_float(curve.first_key);
      float v00y = __int_as_float(curve.num_keys);

      int segment = prim_type >> PRIMITIVE_NUM_TOTAL;

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

        float3 ck0 = hair->curve_keys[ka];

        abvh_leaf.hair.curve[0] = make_float4(ck0.x, ck0.y, ck0.z, hair->curve_radius[ka]);

        float3 cka = hair->curve_keys[k0];

        abvh_leaf.hair.curve[1] = make_float4(cka.x, cka.y, cka.z, hair->curve_radius[k0]);
      }
    }

    encoded_ptr = MAKE_PTR(abvh_cnt, Leaf_Node);
  }

  abvh_node[DECODE_NODE(encoded_ptr)] = abvh_leaf;
  abvh_cnt++;
  return encoded_ptr;
}


uint BVHAMD::flatten_nodes(const BVHNode *node,
                            array<ABVHNode> &abvh_node)
{

    struct BVHStack {
    const BVHNode *node;
    uint child_number;
    uint parent_ptr;

    BVHStack(const BVHNode *in_node, uint in_child_number, uint in_parent_ptr)
        : node(in_node), child_number(in_child_number), parent_ptr(in_parent_ptr)
    {
    }
  };


  uint encoded_ptr = INVALID_NODE;
  uint abvh_cnt = 0;

  std::vector<BVHStack> stack;
  uint qbvh_cnt = 0;
  stack.reserve(BVHParams::MAX_DEPTH * 2);

  stack.push_back(BVHStack(node, INVALID_NODE, INVALID_NODE));

  while (!stack.empty()) {

    BVHStack entry = stack.back();
    stack.pop_back();
    if (entry.node->is_leaf()) {
      encoded_ptr = pack_leaf(entry.node, abvh_node, abvh_cnt);
      uint parent_index = entry.parent_ptr;
      uint child_index = entry.child_number;
      switch (child_index) {
        case 0:
          abvh_node[parent_index].box.child0 = encoded_ptr;
          break;
        case 1:
          abvh_node[parent_index].box.child1 = encoded_ptr;
          break;
        case 2:
          abvh_node[parent_index].box.child2 = encoded_ptr;
          break;
        case 3:
          abvh_node[parent_index].box.child3 = encoded_ptr;
          break;
      }
    }
    else {

      // pack_node

      BVHNode *children[4] = {};
      BVHNode *child0 = entry.node->get_child(0);
      BVHNode *child1 = entry.node->get_child(1);

      BoundBox bounds_[4];

      size_t numChildren = 2;
      children[0] = entry.node->get_child(0);
      bounds_[0] = child0->bounds;
      children[1] = entry.node->get_child(1);
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

      bool valid_nodes[4] = {false, false, false, false};

      for (int i = 0; i < numChildren; i++) {
        if (children[i]) {
          memcpy(&box_f32.children_bound_box[6 * i], &children[i]->bounds.min, 3 * sizeof(float));
          memcpy(&box_f32.children_bound_box[6 * i + 3], &children[i]->bounds.max, 3 * sizeof(float));
          valid_nodes[i] = true;
        }
      }

      encoded_ptr = MAKE_PTR(abvh_cnt, BoxNode32);
      abvh_cnt += 2;
      *(BoxNodeF32 *)(&abvh_node[0] + DECODE_NODE(encoded_ptr)) = box_f32;

      uint parent_index = entry.parent_ptr;
      if (parent_index != INVALID_NODE) {
        uint child_index = entry.child_number;
        switch (child_index) {
          case 0:
            abvh_node[parent_index].box.child0 = encoded_ptr;
            break;
          case 1:
            abvh_node[parent_index].box.child1 = encoded_ptr;
            break;
          case 2:
            abvh_node[parent_index].box.child2 = encoded_ptr;
            break;
          case 3:
            abvh_node[parent_index].box.child3 = encoded_ptr;
            break;
        }
      }

      encoded_ptr = DECODE_NODE(encoded_ptr);
      if (valid_nodes[0])
        stack.push_back(BVHStack(children[0], 0, encoded_ptr));
      else
        abvh_node[encoded_ptr].box.child0 = INVALID_NODE;

      if (valid_nodes[1])
        stack.push_back(BVHStack(children[1], 1, encoded_ptr));
      else
        abvh_node[encoded_ptr].box.child1 = INVALID_NODE;

      if (valid_nodes[2])
        stack.push_back(BVHStack(children[2], 2, encoded_ptr));
      abvh_node[encoded_ptr].box.child2 = INVALID_NODE;

      if (valid_nodes[3])
        stack.push_back(BVHStack(children[3], 3, encoded_ptr));
      else
        abvh_node[encoded_ptr].box.child3 = INVALID_NODE;
    }
  }

  return abvh_cnt;
}


BVHNode *BVHAMD::widen_children_nodes(const BVHNode *root)
{
  return const_cast<BVHNode *>(root);
}

CCL_NAMESPACE_END
