/*
???
 */

/* description
 */

#define BOX_NODE_UPDATE box_node_update
#define LOCAL_INTERSECT local_intersect

// ccl_device_inline
inline __attribute__((always_inline)) bool BVH_FUNCTION_FULL_NAME(ABVH)(
    KernelGlobals *kg,
    const Ray *ray,
    LocalIntersection *local_isect,
    int local_object,
    uint *lcg_state,
    int max_hits)
{
#ifdef LOCAL_STACK
  __local uint *traversal_stack = &(kg->traversal_stack[get_local_id(0)]);
#else
  uint traversal_stack[STACK_SIZE];
#endif
  traversal_stack[0] = INVALID_NODE;
  uint stack_ptr = LOCAL_SIZE;

  uint2 offset = kernel_tex_fetch(__bvh_amd_offset, local_object);
  uint node_offset = offset.x;
  uint prim_offset = offset.y;
  uint nodeIdx = BoxNode32;

  float3 P = ray->P;
  float3 dir = bvh_clamp_direction(ray->D);
  float3 idir = bvh_inverse_direction(dir);
  int object = OBJECT_NONE;
  float isect_t = ray->t;

  if (local_isect != NULL) {
    local_isect->num_hits = 0;
  }

  const int object_flag = kernel_tex_fetch(__object_flag, local_object);
  if (!(object_flag & SD_OBJECT_TRANSFORM_APPLIED)) {
#if BVH_FEATURE(BVH_MOTION)
    Transform ob_itfm;
    isect_t = bvh_instance_motion_push(kg, local_object, ray, &P, &dir, &idir, isect_t, &ob_itfm);
#else
    isect_t = bvh_instance_push(kg, local_object, ray, &P, &dir, &idir, isect_t);
#endif
    object = local_object;
  }

  __global Node const *restrict nodes = (__global Node *)(&kernel_tex_fetch(__bvh_amd,
                                                                            node_offset));

  while (nodeIdx != INVALID_NODE) {
    uint node_type = NODE_TYPE(nodeIdx);

    ulong2 bvh_desc = pack_bvh_descriptor(nodes, -1ul, 1u, 6u, 0u, 1, 0u);
    uint4 res = image_bvh_intersect_ray(nodeIdx, isect_t, P, dir, idir, bvh_desc);

    if (node_type == Leaf_Node) {

      TriangleNode_ *tri = (TriangleNode_ *)(nodes + ((nodeIdx) >> 3u));

      int prim_adr = tri->shape_id + prim_offset;
      int type = tri->prim_type;
      switch (type & PRIMITIVE_ALL) {
        case PRIMITIVE_TRIANGLE: {

          if (local_intersect(kg,
                              tri,
                              local_isect,
                              res,
                              object,
                              local_object,
                              prim_adr,
                              isect_t,
                              lcg_state,
                              max_hits)) {
            return true;
          }

          break;
        }
#if BVH_FEATURE(BVH_MOTION)
        case PRIMITIVE_MOTION_TRIANGLE: {
          if (motion_tri_intersect_local(kg,
                                         local_isect,
                                         tri,
                                         P,
                                         dir,
                                         ray->time,
                                         object,
                                         local_object,
                                         prim_addr,
                                         isect_t,
                                         lcg_state,
                                         max_hits)) {
            return true;
          }
          break;
        }
#endif
        default: {
          break;
        }
      }
    }
    else {
      if (BOX_NODE_UPDATE(&traversal_stack[0], &stack_ptr, res, false, &nodeIdx))
        continue;
    }

    stack_ptr -= LOCAL_SIZE;
    nodeIdx = traversal_stack[stack_ptr];
  }

  return false;
}

#undef BOX_NODE_UPDATE
#undef LOCAL_INTERSECT
