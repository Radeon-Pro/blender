#ifndef _BVH_AMD_H_
#define _BVH_AMD_H_

#include "bvh/bvh.h"
#include "util/util_map.h"
//#include "render/object.h"

CCL_NAMESPACE_BEGIN

#define INVALID_NODE 0xffffffffU


struct EmbreeNode;
struct EmbreeCompactNode;


class Mesh;
class Hair;
class BVHNode;
class BVHObjectBinning;





struct Root_Node {
  BVHNode *bvh2;
  EmbreeNode *embree;
};

struct Unaligned_BoxNode {

  uint child[4];

  Transform aligned_space0[2];

  uint transform_id;
  uint object_id;
  uint aligned;
  uint visibility;

  Transform aligned_space1[2];

  uint padding[8];
};

struct BoxNodeF32 { 

  uint child[4];

  float children_bound_box[24];

  uint prim_offset;
  uint object_id;
  uint aligned;
  uint visibility;
};

struct ObjectNode {

  Transform tfm;
  uint shape_id;
  uint unused[2];
  uint used;

};

struct TriangleNode {

  float vertices[9];
  int unused[3];
  uint shape_id;
  uint prim_type;
  uint prim_visibility;
  uint triangle_id;


  TriangleNode(float3 v0, float3 v1, float3 v2)
  {
    vertices[0] = v0.x;
    vertices[1] = v0.y;
    vertices[2] = v0.z;

    vertices[3] = v1.x;
    vertices[4] = v1.y;
    vertices[5] = v1.z;

    vertices[6] = v2.x;
    vertices[7] = v2.y;
    vertices[8] = v2.z;
  }

};

struct HairNode {

  int prim_object;
  int prim_idx;
  int k1;
  int kb;
  float4 curve[2];
  uint used[4];

};


struct ABVHNode {
  union {
    ObjectNode object;
    TriangleNode tri;
    HairNode hair;
  };

    ABVHNode()
    {
      memset(this, 0, sizeof(ABVHNode));
    }
};

class BVHAMD : public BVH {

  public:
  void build(Progress &progress, Stats *stats);
  //void refit_nodes(Progress &progress);

  std::vector<BVHReference> reference;
  std::vector<Geometry *> geometry_list;
  PackedBVH pack;

 protected:
  friend class BVH;
  BVHAMD(const BVHParams &params,
         const vector<Geometry *> &geometry,
         const vector<Object *> &objects
	  //,const Device *device
    );

  /*virtual ~BVHAMD()
  {
  }*/
  //virtual void copy_to_device(Progress &progress, DeviceScene *dscene) override;
  void pack_nodes(const BVHNode *root);
  void pack_instances();
  void pack_primitives();
  void pack_triangle(int idx, float4 tri_verts[3]);

  uint flatten_nodes(const BVHNode *root, uint &qbvh_cnt);

  uint pack_leaf(void *node, uint &abvh_cnt);

  

  //virtual BVHNode *widen_children_nodes(const BVHNode *) override;



  enum Node_Type {
    Triangle_Node = 0,
    Motion_Triangle_Node = 1,
    BoxNode16 = 4,
    BoxNode32 = 5,
    Object_Node = 6,
    Hair_Node = 7
  };



  std::map<Geometry *, uint> geometry_users;
  BoundBox root_bounds, center;
  std::vector<RTCBuildPrimitive> embree_primitives;// change to array
  bool is_embree;





  Root_Node createEmbreePrimitive(Progress &progress); //breakdown and clean-up 
  void add_reference_geometry(BoundBox &root, BoundBox &center, Geometry *geom, int i, Object *ob);
  void add_reference_triangles(BoundBox &root, BoundBox &center, Mesh *mesh, int i, Object *ob);
  void add_reference_curves(BoundBox &root, BoundBox &center, Hair *hair, int i, Object *ob);
  void add_reference_object(BoundBox &root, BoundBox &center, Object *ob, int i);
  void add_reference(Object *ob, int i);

  //consolidate with flatten_node and pack_node
  uint pack_node_embree(EmbreeCompactNode *node,
                        uint parent_index,
                        uint &abvh_cnt,
                        bool **valid_nodes,
                        EmbreeCompactNode **children);
  void flatten_nodes_embree(EmbreeNode *node, uint &abvh_cnt);

    array<uint> primitives_offset;

};

CCL_NAMESPACE_END

#endif
