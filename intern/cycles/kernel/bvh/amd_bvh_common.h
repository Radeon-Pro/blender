
#ifdef __AMD_RT_HWI__


#  if defined(__AMD_LOCAL)
#    define LOCAL_STACK
#  endif

#  ifdef LOCAL_STACK
#    define LOCAL_SIZE 64
#    define LOCAL_STACK_SIZE 24
#    define STACK_SIZE LOCAL_SIZE *LOCAL_STACK_SIZE
#  else
#    define LOCAL_SIZE 1
#    define STACK_SIZE 384
#  endif

typedef struct {
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
} TriangleNode_;


typedef struct {

  int prim_object;
  int prim_idx;
  int k1;
  int kb;
  float4 curve[2];
  uint used[4];
} HairNode;


#include "traversal.h"

enum Node_Type { Leaf_Node = 0, BoxNode16 = 4, BoxNode32 = 5, Object_Node = 6 };



ccl_device_inline bool local_intersect(KernelGlobals *kg,
                                       TriangleNode_ *triangle_node,
                                       LocalIntersection *local_isect,
                                       uint4 res,
                                       int object,
                                       int local_object,
                                       int prim_adr,
                                       float isect_t,
                                       uint *lcg_state,
                                       int max_hits)
{

  /*if (object == OBJECT_NONE) {
    int prim_object = __float_as_int(triangle_node->data0);
    if (prim_object != local_object) {//can we embede this in the node?
      return false;
    }
  }*/
  if (object == OBJECT_NONE) {
    if (kernel_tex_fetch(__prim_object, prim_adr) != local_object) {
      return false;
    }
  }

  float det = as_float(res.y);
  float inv_denom = native_recip(det);
  float f = as_float(res.x) * inv_denom;

  if (f >= isect_t) {
    return false;
  }

  if (max_hits == 0) {
    return true;
  }

  int hit;
  if (lcg_state) {
    for (int i = min(max_hits, local_isect->num_hits) - 1; i >= 0; --i) {
      if (local_isect->hits[i].t == f) {
        return false;
      }
    }

    local_isect->num_hits++;

    if (local_isect->num_hits <= max_hits) {
      hit = local_isect->num_hits - 1;
    }
    else {
      hit = lcg_step_uint(lcg_state) % local_isect->num_hits;

      if (hit >= max_hits)
        return false;
    }
  }
  else {

    if (local_isect->num_hits && f > local_isect->hits[0].t) {
      return false;
    }

    hit = 0;
    local_isect->num_hits = 1;
  }

  /* Record intersection. */
  Intersection *isect = &local_isect->hits[hit];
  isect->prim = prim_adr;
  isect->object = object;
  isect->type = PRIMITIVE_TRIANGLE;
  isect->u = as_float(res.z) * inv_denom;
  isect->v = as_float(res.w) * inv_denom;
  ;
  isect->t = f;

  const float3 tri_a = (float3)(triangle_node->v0x, triangle_node->v0y, triangle_node->v0z);
  const float3 tri_b = (float3)(triangle_node->v1x, triangle_node->v1y, triangle_node->v1z);
  const float3 tri_c = (float3)(triangle_node->v2x, triangle_node->v2y, triangle_node->v2z);

  local_isect->Ng[hit] = normalize(cross(tri_b - tri_a, tri_c - tri_a));

  return false;
}

//#if BVH_FEATURE(BVH_HAIR)
#  if defined(__HAIR__)
ccl_device_forceinline bool hair_intersect(KernelGlobals *kg,
                                           HairNode *hair,
                                           Intersection *isect,
                                           int object,
                                           const float3 P,
                                           const float3 dir,
                                           float time,
                                           int type)

{
  const bool is_motion = (type & PRIMITIVE_ALL_MOTION);

  if (is_motion && kernel_data.bvh.use_bvh_steps) {
    if (time < hair->curve[0].z || time > hair->curve[0].w)
      return false;
  }
  else {
    float4 curve[4];
    if (!is_motion) {
      curve[0] = hair->curve[0];
      curve[1] = hair->curve[1];
      curve[2] = kernel_tex_fetch(__curve_keys, hair->k1);
      curve[3] = kernel_tex_fetch(__curve_keys, hair->kb);
    }
    else {
      int prim = hair->prim_idx;
      int fobject = (object == OBJECT_NONE) ? hair->prim_object : object;
      int k0 = __float_as_int(hair->curve[0].x);
      int ka = __float_as_int(hair->curve[0].y);
      motion_curve_keys(kg, fobject, prim, time, ka, k0, hair->k1, hair->kb, curve);
    }

    if (type & (PRIMITIVE_CURVE_RIBBON | PRIMITIVE_MOTION_CURVE_RIBBON)) {

      const int subdivisions = kernel_data.bvh.curve_subdivisions;
      if (ribbon_intersect(P, dir, isect->t, subdivisions, curve, isect))
        return true;
    }
    else if (curve_intersect_recursive(P, dir, curve, isect))
      return true;
  }

  return false;
}
#  endif

ccl_device_forceinline bool tri_intersect_(float4 res, Intersection *isect)
{
  float inv_denom = native_recip(res.y);
  float f = res.x * inv_denom;

  if (f < isect->t) {

    isect->u = res.z * inv_denom;
    isect->v = res.w * inv_denom;
    isect->t = f;
    return true;
  }
  return false;
}

#  if defined(__OBJECT_MOTION__)
ccl_device_inline void get_motion_triangle_verts_for_step(KernelGlobals *kg,
                                                          TriangleNode *triangle_node,
                                                          uint4 tri_vindex,
                                                          int offset,
                                                          int numverts,
                                                          int numsteps,
                                                          int step,
                                                          float3 verts[3])
{
  if (step == numsteps) {
    verts[0] = (float3)(triangle_node->v0x, triangle_node->v0y, triangle_node->v0z);
    verts[1] = (float3)(triangle_node->v1x, triangle_node->v1y, triangle_node->v1z);
    verts[2] = (float3)(triangle_node->v2x, triangle_node->v2y, triangle_node->v2z);
  }
  else {
    if (step > numsteps)
      step--;

    offset += step * numverts;

    verts[0] = float4_to_float3(kernel_tex_fetch(__attributes_float3, offset + tri_vindex.x));
    verts[1] = float4_to_float3(kernel_tex_fetch(__attributes_float3, offset + tri_vindex.y));
    verts[2] = float4_to_float3(kernel_tex_fetch(__attributes_float3, offset + tri_vindex.z));
  }
}

ccl_device_inline bool motion_tri_intersect(KernelGlobals *kg,
                                            Intersection *isect,
                                            TriangleNode *triangle_node,
                                            float3 P,
                                            float3 dir,
                                            float time,
                                            int object,
                                            int prim_adr)
{

  int prim = kernel_tex_fetch(__prim_index, prim_adr);
  int fobject = (object == OBJECT_NONE) ? kernel_tex_fetch(__prim_object, prim_adr) : object;

  int numsteps = kernel_tex_fetch(__objects, object).numsteps;
  int numverts = kernel_tex_fetch(__objects, object).numverts;

  int maxstep = numsteps * 2;
  int step = min((int)(time * maxstep), maxstep - 1);
  float t = time * maxstep - step;

  AttributeElement elem;
  int offset = find_attribute_motion(kg, object, ATTR_STD_MOTION_VERTEX_POSITION, &elem);

  float3 verts[3];
  float3 next_verts[3];

  uint4 tri_vindex = kernel_tex_fetch(__tri_vindex, prim);

  get_motion_triangle_verts_for_step(
      kg, triangle_node, tri_vindex, offset, numverts, numsteps, step, verts);
  get_motion_triangle_verts_for_step(
      kg, triangle_node, tri_vindex, offset, numverts, numsteps, step + 1, next_verts);

  verts[0] = (1.0f - t) * verts[0] + t * next_verts[0];
  verts[1] = (1.0f - t) * verts[1] + t * next_verts[1];
  verts[2] = (1.0f - t) * verts[2] + t * next_verts[2];

  float f, u, v;
  if (ray_triangle_intersect(P, dir, isect->t, verts[0], verts[1], verts[2], &u, &v, &f)) {
    isect->t = f;
    isect->u = u;
    isect->v = v;
    return true;
  }
  return false;
}

#    ifdef __BVH_LOCAL__
ccl_device_inline bool motion_tri_intersect_local(KernelGlobals *kg,
                                                  LocalIntersection *local_isect,
                                                  TriangleNode *triangle_node,
                                                  float3 P,
                                                  float3 dir,
                                                  float time,
                                                  int object,
                                                  int local_object,
                                                  int prim_adr,
                                                  float tmax,
                                                  uint *lcg_state,
                                                  int max_hits)
{

  if (object == OBJECT_NONE) {
    if (kernel_tex_fetch(__prim_object, prim_adr) != local_object) {
      return false;
    }
  }

  int prim = kernel_tex_fetch(__prim_index, prim_adr);
  int fobject = (object == OBJECT_NONE) ? kernel_tex_fetch(__prim_object, prim_adr) : object;

  int numsteps = kernel_tex_fetch(__objects, object).numsteps;
  int numverts = kernel_tex_fetch(__objects, object).numverts;

  int maxstep = numsteps * 2;
  int step = min((int)(time * maxstep), maxstep - 1);
  float t = time * maxstep - step;

  AttributeElement elem;
  int offset = find_attribute_motion(kg, object, ATTR_STD_MOTION_VERTEX_POSITION, &elem);

  float3 verts[3];
  float3 next_verts[3];

  /*get_motion_triangle_verts_for_step(kg, triangle_node, offset, numverts, numsteps, step, verts);
  get_motion_triangle_verts_for_step(
      kg, triangle_node, offset, numverts, numsteps, step + 1, next_verts);*/
  uint4 tri_vindex = kernel_tex_fetch(__tri_vindex, prim);

  get_motion_triangle_verts_for_step(
      kg, triangle_node, tri_vindex, offset, numverts, numsteps, step, verts);
  get_motion_triangle_verts_for_step(
      kg, triangle_node, tri_vindex, offset, numverts, numsteps, step + 1, next_verts);

  verts[0] = (1.0f - t) * verts[0] + t * next_verts[0];
  verts[1] = (1.0f - t) * verts[1] + t * next_verts[1];
  verts[2] = (1.0f - t) * verts[2] + t * next_verts[2];

  float f, u, v;
  if (!ray_triangle_intersect(P, dir, isect->t, verts[0], verts[1], verts[2], &u, &v, &f))
    return false;

  if (max_hits == 0) {
    return true;
  }

  int hit;
  if (lcg_state) {
    for (int i = min(max_hits, local_isect->num_hits) - 1; i >= 0; --i) {
      if (local_isect->hits[i].t == f) {
        return false;
      }
    }

    local_isect->num_hits++;

    if (local_isect->num_hits <= max_hits) {
      hit = local_isect->num_hits - 1;
    }
    else {
      hit = lcg_step_uint(lcg_state) % local_isect->num_hits;

      if (hit >= max_hits)
        return false;
    }
  }
  else {

    if (local_isect->num_hits && t > local_isect->hits[0].t) {
      return false;
    }

    hit = 0;
    local_isect->num_hits = 1;
  }

  Intersection *isect = &local_isect->hits[hit];
  isect->t = f;
  isect->u = u;
  isect->v = v;
  isect->prim = prim_adr;
  isect->object = object;
  isect->type = PRIMITIVE_MOTION_TRIANGLE;

  local_isect->Ng[hit] = normalize(cross(verts[1] - verts[0], verts[2] - verts[0]));

  return false;
}
#    endif

#  endif

#endif
