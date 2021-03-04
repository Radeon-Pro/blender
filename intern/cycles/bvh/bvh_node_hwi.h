#pragma once
#include <cstdint>

#define MAX_DATA_SIZE 124
#define PADDING_SIZE 256

#define MAKE_PTR(node_addr, node_type) (((node_type)&7u) | ((node_addr) << 3u))
#define NODE_TYPE(A) (((A)) & 7u)
#define DECODE_NODE(ind) ((ind) >> 3u)

typedef unsigned char uchar;
typedef unsigned int uint;
typedef unsigned short ushort;

struct TriangleNode
{
    float v0[3];
    float v1[3];
    float v2[3];
    float v3[3];

    cl_uint primitive_id;  // triangles_count;
    cl_uint padding0;
    cl_uint padding1;

    cl_uint id;
};

struct BoxNodeF16
{
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

struct BoxNodeF32
{
    uint child0;
    uint child1;
    uint child2;
    uint child3;

    float box0_min_x;
    float box0_min_y;
    float box0_min_z;
    float box0_max_x;
    float box0_max_y;
    float box0_max_z;

    float box1_min_x;
    float box1_min_y;
    float box1_min_z;
    float box1_max_x;
    float box1_max_y;
    float box1_max_z;

    float box2_min_x;
    float box2_min_y;
    float box2_min_z;
    float box2_max_x;
    float box2_max_y;
    float box2_max_z;

    float box3_min_x;
    float box3_min_y;
    float box3_min_z;
    float box3_max_x;
    float box3_max_y;
    float box3_max_z;

    uint padding0;
    uint padding1;
    uint padding2;
    uint padding3;
};
/*
struct BoxNodeF32 {

  uint child[4];

  float children_bound_box[24];

  uint prim_offset;
  uint object_id;
  uint aligned;
  uint visibility;
};*/

struct ObjectNode
{
    cl_amd_transform tfm;
    cl_uint shape_id;
    cl_uint bottom_pointer;
    cl_uint prim_offset;
    cl_uint padding1;
};

struct CustomNode
{
    uint size;
    char data[MAX_DATA_SIZE];
};

struct ABVHNode
{
    union
    {
        BoxNodeF16 box;
        ObjectNode object;
        TriangleNode tri;
        
    };
};

enum BVH_STAT { BVH_STAT_NODE_COUNT, BVH_STAT_LEAF_COUNT };

inline TriangleNode convert_to_gpu_triangle(cl_amd_triangle triangle)
{
    TriangleNode gpu_triangle;
    gpu_triangle.v0[0] = triangle.v0.x;
    gpu_triangle.v0[1] = triangle.v0.y;
    gpu_triangle.v0[2] = triangle.v0.z;

    gpu_triangle.v1[0] = triangle.v1.x;
    gpu_triangle.v1[1] = triangle.v1.y;
    gpu_triangle.v1[2] = triangle.v1.z;

    gpu_triangle.v2[0] = triangle.v2.x;
    gpu_triangle.v2[1] = triangle.v2.y;
    gpu_triangle.v2[2] = triangle.v2.z;
    gpu_triangle.primitive_id = triangle.id;

    return gpu_triangle;
}
