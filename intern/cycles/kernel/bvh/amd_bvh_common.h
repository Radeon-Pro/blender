
#ifdef __AMD_RT_HWI__

#define SORT(childA,childB,distA,distB) if((childB!=INVALID_NODE&&distB<distA)||childA==INVALID_NODE){  float t0 = distA; uint t1 = childA;  childA = childB; distA = distB;  childB=t1; distB=t0; }

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

  int data0;
  int data1;
  int data2;

  uint shape_id;
  uint prim_type;
  uint prim_visibility;
  uint triangle_id;
} TriangleNode;

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

#  define TOP_LEVEL 0x80000000
#  define GET_NODE(node) ((node & ~TOP_LEVEL) >> 3u)

#  define NODE_TYPE(A) ((A)&7u)
#  define INVALID_NODE 0xffffffff

typedef struct {

  Transform tfm;
  uint data_used[4];

} ObjectNode;

typedef struct {

  int prim_object;
  int prim_idx;
  int k1;
  int kb;
  float4 curve[2];
  uint used[4];
} HairNode;

typedef struct {
  uint data[16];
} Node;

typedef struct {
  uint child0;
  uint child1;
  uint child2;
  uint child3;
  
  float bbox[24];


  uint prim_offset;
  uint padding1;
  uint padding2;
  uint padding3;

} BoxNodeF32;


typedef struct {

  uint child0;
  uint child1;
  uint child2;
  uint child3;

  Transform aligned_space0[2];

  uint transform_id;
  uint object_id;
  uint aligned;
  uint visibility;

  Transform aligned_space1[2];

  uint padding[8];
}Unaligned_BoxNode;

enum Node_Type { Leaf_Node = 0, BoxNode16 = 4, BoxNode32 = 5, Object_Node = 6 };

ulong2 pack_bvh_descriptor(const global Node *bvh_base,
                           ulong size,
                           uint box_sort_en,
                           uint box_grow_ulp,
                           uint big_page,
                           uint triangle_return_mode,
                           uint llc_noalloc)
{

  ulong base_address = ((const global uchar *)bvh_base) - ((const global uchar *)0x0);
  base_address = (base_address / 256UL) & 0xffffffffffUL;
  size &= 0x3ffffffffffUL;
  box_grow_ulp &= 0xff;
  box_sort_en &= 1u;
  big_page &= 1u;
  triangle_return_mode &= 1;
  llc_noalloc &= 0x3;
  ulong type = 0x8;

  ulong2 descriptor = (ulong2)(0, 0);

  descriptor.s0 = base_address | ((ulong)(box_grow_ulp) << 55ul) | ((ulong)(box_sort_en) << 63ul);
  descriptor.s1 = size | ((ulong)(big_page) << 59ul) | ((ulong)(type) << 60ul) |
                  ((ulong)(triangle_return_mode) << 56ul) | ((ulong)(llc_noalloc) << 57ul);

  return descriptor;
}

__attribute__((always_inline)) bool box_node_update(
#  ifdef LOCAL_STACK
    __local
#  endif
        uint *traversal_stack,
    uint *ptr,
    uint4 res,
    uint top_level,
    uint *nodeIdx)
{
  if (res.w != INVALID_NODE | res.z != INVALID_NODE | res.y != INVALID_NODE |
      res.x != INVALID_NODE) {

    uint flag = top_level ? TOP_LEVEL : 0x0;
    int stack_ptr = *ptr;

    if (stack_ptr < STACK_SIZE && res.w != INVALID_NODE) {

      traversal_stack[stack_ptr] = res.w | flag;
      stack_ptr += LOCAL_SIZE;
    }

    if (stack_ptr < STACK_SIZE && res.z != INVALID_NODE) {

      traversal_stack[stack_ptr] = res.z | flag;
      stack_ptr += LOCAL_SIZE;
    }

    if (stack_ptr < STACK_SIZE && res.y != INVALID_NODE) {

      traversal_stack[stack_ptr] = res.y | flag;
      stack_ptr += LOCAL_SIZE;
    }

    *ptr = stack_ptr;

    *nodeIdx = (res.x | flag);
    return true;
  }
  return false;
}

//ccl_device_inline uint4 unaligned_box_intersect(KernelGlobals *kg, __global BoxNodeF32* box_node, uint transform_id, float3 dir, float3 P, float extent)
ccl_device_inline uint4 unaligned_box_intersect(Unaligned_BoxNode* box_node, float3 dir, float3 P, float extent)
{
	/*Transform aligned_space0 = *(__global Transform*)(&kernel_tex_fetch(__aligned_space, transform_id));
	Transform aligned_space1 = *(__global Transform*)(&kernel_tex_fetch(__aligned_space, transform_id + 1));
	Transform aligned_space2 = *(__global Transform*)(&kernel_tex_fetch(__aligned_space, transform_id + 2));
	Transform aligned_space3 = *(__global Transform*)(&kernel_tex_fetch(__aligned_space, transform_id + 3));*/
	
	Transform aligned_space0 = box_node->aligned_space0[0];
	Transform aligned_space1 = box_node->aligned_space0[1];
	Transform aligned_space2 = box_node->aligned_space1[0];
	Transform aligned_space3 = box_node->aligned_space1[1];


	
	float4 tfm_x_x = (float4)(aligned_space0.x.x, aligned_space1.x.x, aligned_space2.x.x, aligned_space3.x.x);
	float4 tfm_x_y = (float4)(aligned_space0.x.y, aligned_space1.x.y, aligned_space2.x.y, aligned_space3.x.y);
	float4 tfm_x_z = (float4)(aligned_space0.x.z, aligned_space1.x.z, aligned_space2.x.z, aligned_space3.x.z);
	
	float4 tfm_y_x = (float4)(aligned_space0.y.x, aligned_space1.y.x, aligned_space2.y.x, aligned_space3.y.x);
	float4 tfm_y_y = (float4)(aligned_space0.y.y, aligned_space1.y.y, aligned_space2.y.y, aligned_space3.y.y);
	float4 tfm_y_z = (float4)(aligned_space0.y.z, aligned_space1.y.z, aligned_space2.y.z, aligned_space3.y.z);
	
	float4 tfm_z_x = (float4)(aligned_space0.z.x, aligned_space1.z.x, aligned_space2.z.x, aligned_space3.z.x);
	float4 tfm_z_y = (float4)(aligned_space0.z.y, aligned_space1.z.y, aligned_space2.z.y, aligned_space3.z.y);
	float4 tfm_z_z = (float4)(aligned_space0.z.z, aligned_space1.z.z, aligned_space2.z.z, aligned_space3.z.z);
	
	
	float4 tfm_t_x = (float4)(aligned_space0.x.w, aligned_space1.x.w, aligned_space2.x.w, aligned_space3.x.w);
	float4 tfm_t_y = (float4)(aligned_space0.y.w, aligned_space1.y.w, aligned_space2.y.w, aligned_space3.y.w);
	float4 tfm_t_z = (float4)(aligned_space0.z.w, aligned_space1.z.w, aligned_space2.z.w, aligned_space3.z.w);
	
	
	const float4 aligned_dir_x = dir.x * tfm_x_x + dir.y * tfm_x_y + dir.z * tfm_x_z,
				 aligned_dir_y = dir.x * tfm_y_x + dir.y * tfm_y_y + dir.z * tfm_y_z,
				 aligned_dir_z = dir.x * tfm_z_x + dir.y * tfm_z_y + dir.z * tfm_z_z;
				 
	const float4 aligned_P_x = P.x * tfm_x_x + P.y * tfm_x_y + P.z * tfm_x_z + tfm_t_x,
	 aligned_P_y = P.x * tfm_y_x + P.y * tfm_y_y + P.z * tfm_y_z + tfm_t_y,
	 aligned_P_z = P.x * tfm_z_x + P.y * tfm_z_y + P.z * tfm_z_z + tfm_t_z;
	 
	 
	 
	 
	 const float4 neg_one = (float4)(-1.0f, -1.0f, -1.0f, -1.0f);
	  const float4 nrdir_x = neg_one / aligned_dir_x, nrdir_y = neg_one / aligned_dir_y,
				 nrdir_z = neg_one / aligned_dir_z;

	  const float4 tlower_x = aligned_P_x * nrdir_x, tlower_y = aligned_P_y * nrdir_y,
				 tlower_z = aligned_P_z * nrdir_z;

	  const float4 tupper_x = tlower_x - nrdir_x, tupper_y = tlower_y - nrdir_y,
				 tupper_z = tlower_z - nrdir_z;


	  const float4 tnear_x = min(tlower_x, tupper_x);
	  const float4 tnear_y = min(tlower_y, tupper_y);
	  const float4 tnear_z = min(tlower_z, tupper_z);
	  const float4 tfar_x = max(tlower_x, tupper_x);
	  const float4 tfar_y = max(tlower_y, tupper_y);
	  const float4 tfar_z = max(tlower_z, tupper_z);
	  const float4 t_near = max(max(0, tnear_x), max(tnear_y, tnear_z));
	  const float4 t_far = min(min(extent, tfar_x), min(tfar_y, tfar_z));
	
	
	  float2 s0 = (float2)(t_near.x, t_far.x);	
	  float2 s1 = (float2)(t_near.y, t_far.y);
	  float2 s2 = (float2)(t_near.z, t_far.z);
	  float2 s3 = (float2)(t_near.w, t_far.w);			  

	
	uint traverse0 = t_near.x <= t_far.x ? box_node->child0 : INVALID_NODE;
	uint traverse1 = t_near.y <= t_far.y ? box_node->child1 : INVALID_NODE;
	uint traverse2 = t_near.z <= t_far.z ? box_node->child2 : INVALID_NODE;
	uint traverse3 = t_near.w <= t_far.w ? box_node->child3 : INVALID_NODE;

	
	SORT(traverse0,traverse2,s0.x,s2.x)
	SORT(traverse1,traverse3,s1.x,s3.x)
	SORT(traverse0,traverse1,s0.x,s1.x)
	SORT(traverse2,traverse3,s2.x,s3.x)
	SORT(traverse1,traverse2,s1.x,s2.x)			

	uint4 res= (uint4)(traverse0,traverse1,traverse2,traverse3);
	return res;
}

ccl_device_inline bool local_intersect(KernelGlobals *kg,
                                       TriangleNode *triangle_node,
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

ccl_device_forceinline bool tri_intersect(float4 res, Intersection *isect)
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

__attribute__((always_inline, pure, overloadable)) uint4 __llvm_amdgcn_image_bvh_intersect_ray(
    uint node_ptr,
    float ray_extent,
    float4 ray_origin,
    float4 ray_dir,
    float4 ray_inv_dir,
    uint4 texture_descr) __asm("llvm.amdgcn.image.bvh.intersect.ray.i32.v4f32");
__attribute__((always_inline, pure, overloadable)) uint4 __llvm_amdgcn_image_bvh_intersect_ray(
    uint node_ptr,
    float ray_extent,
    float4 ray_origin,
    half4 ray_dir,
    half4 ray_inv_dir,
    uint4 texture_descr) __asm("llvm.amdgcn.image.bvh.intersect.ray.i32.v4f16");
__attribute__((always_inline, pure, overloadable)) uint4 __llvm_amdgcn_image_bvh_intersect_ray(
    ulong node_ptr,
    float ray_extent,
    float4 ray_origin,
    float4 ray_dir,
    float4 ray_inv_dir,
    uint4 texture_descr) __asm("llvm.amdgcn.image.bvh.intersect.ray.i64.v4f32");
__attribute__((always_inline, pure, overloadable)) uint4 __llvm_amdgcn_image_bvh_intersect_ray(
    ulong node_ptr,
    float ray_extent,
    float4 ray_origin,
    half4 ray_dir,
    half4 ray_inv_dir,
    uint4 texture_descr) __asm("llvm.amdgcn.image.bvh.intersect.ray.i64.v4f16");

inline uint4 image_bvh_intersect_ray(uint node_addr,
                                     float ray_extent,
                                     float3 ray_origin,
                                     float3 ray_dir,
                                     float3 ray_inv_dir,
                                     const ulong2 bvh_descriptor)
{
  return __llvm_amdgcn_image_bvh_intersect_ray(node_addr,
                                               ray_extent,
                                               ray_origin.xyzz,
                                               ray_dir.xyzz,
                                               ray_inv_dir.xyzz,
                                               as_uint4(bvh_descriptor));
}

inline uint4 image_bvh64_intersect_ray(ulong node_addr,
                                       float ray_extent,
                                       float3 ray_origin,
                                       float3 ray_dir,
                                       float3 ray_inv_dir,
                                       const ulong2 bvh_descriptor)
{
  return __llvm_amdgcn_image_bvh_intersect_ray(node_addr,
                                               ray_extent,
                                               ray_origin.xyzz,
                                               ray_dir.xyzz,
                                               ray_inv_dir.xyzz,
                                               as_uint4(bvh_descriptor));
}

#endif
