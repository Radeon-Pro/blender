#ifndef _BVH_AMD_H_
#define _BVH_AMD_H_

#include "bvh/bvh.h"
#include "bvh/bvh_params.h"

CCL_NAMESPACE_BEGIN

struct TriangleNode {
  float v0x;
  float v0y;
  float v0z;

  float v1x;
  float v1y;
  float v1z;

  float v2x;
  float v2y;
  float v2z;

  float data0;
  float data1;
  float data2;

  uint shape_id;
  uint prim_type;
  uint prim_visibility;
  uint triangle_id;
};

struct BoxNodeF16 {
  uint child0;
  uint child1;
  uint child2;
  uint child3;

  uint16_t box0_min_x;
  uint16_t box0_min_y;
  uint16_t box0_min_z;
  uint16_t box0_max_x;
  uint16_t box0_max_y;
  uint16_t box0_max_z;

  uint16_t box1_min_x;
  uint16_t box1_min_y;
  uint16_t box1_min_z;
  uint16_t box1_max_x;
  uint16_t box1_max_y;
  uint16_t box1_max_z;

  uint16_t box2_min_x;
  uint16_t box2_min_y;
  uint16_t box2_min_z;
  uint16_t box2_max_x;
  uint16_t box2_max_y;
  uint16_t box2_max_z;

  uint16_t box3_min_x;
  uint16_t box3_min_y;
  uint16_t box3_min_z;
  uint16_t box3_max_x;
  uint16_t box3_max_y;
  uint16_t box3_max_z;
};

struct BoxNodeF32 {
  uint child0;
  uint child1;
  uint child2;
  uint child3;

  float children_bound_box[24];

  uint padding0;
  uint padding1;
  uint padding2;
  uint padding3;
};

struct ObjectNode {

  Transform tfm;
  uint used[4];
};

struct HairNode {

  int prim_object;
  int prim_idx;
  int k1;
  int kb;
  float4 curve[2];
  uint used[4];
};

struct bbox {
  float3 pmin;
  float3 pmax;
};

struct ABVHNode {
  union {
    BoxNodeF16 box;
    ObjectNode object;
    TriangleNode tri;
    HairNode hair;
  };
};

class BVHNode;

class BVHAMD : public BVH {

 protected:
  friend class BVH;
  BVHAMD(const BVHParams &params,
         const vector<Geometry *> &geometry,
         const vector<Object *> &objects);

  virtual ~BVHAMD()
  {
  }
  virtual void copy_to_device(Progress &progress, DeviceScene *dscene) override;
  void pack_nodes(const BVHNode *root) override;
  void pack_instances(size_t nodes_size, size_t leaf_nodes_size);

  uint flatten_nodes(const BVHNode *root, array<ABVHNode> &node);
  uint pack_leaf(const BVHNode *node, array<ABVHNode> &abvh_node, uint &abvh_cnt);

  virtual void refit_nodes() override;

  virtual BVHNode *widen_children_nodes(const BVHNode *) override;

  enum Node_Type { Leaf_Node = 0, BoxNode16 = 4, BoxNode32 = 5, Object_Node = 6 };
};

CCL_NAMESPACE_END

#endif
