/*

 */
#define BOX_NODE_UPDATE box_node_update

inline __attribute__((always_inline)) bool BVH_FUNCTION_FULL_NAME(ABVH)(KernelGlobals *kg,
                                                                        const Ray *ray,
                                                                        Intersection *isect,
                                                                        const uint visibility)
{
#ifdef LOCAL_STACK
  __local uint *traversal_stack = &(kg->traversal_stack[get_local_id(0)]);
#else
  uint traversal_stack[STACK_SIZE];
#endif

  traversal_stack[0] = INVALID_NODE;

  uint stack_ptr = LOCAL_SIZE;

  float3 P = ray->P;
  float3 dir = bvh_clamp_direction(ray->D);
  float3 idir = bvh_inverse_direction(dir);
  int object = OBJECT_NONE;

  float trnasform_factor;

  isect->t = ray->t;
  isect->u = 0.0f;
  isect->v = 0.0f;
  isect->prim = PRIM_NONE;
  isect->object = OBJECT_NONE;

  uint nodeIdx = BoxNode32 | TOP_LEVEL;

  uint prim_offset = 0;
  uint triangle_return_mode = 1;

  __global Node const *restrict root = (__global Node *)(&kernel_tex_fetch(__bvh_amd, 0));
  __global Node const *restrict nodes = root;

#if BVH_FEATURE(BVH_MOTION)
  Transform ob_itfm;
#endif

  while (nodeIdx != INVALID_NODE) {

    uint top_level = nodeIdx & TOP_LEVEL;

    nodeIdx &= (~TOP_LEVEL);

    ulong2 bvh_desc = pack_bvh_descriptor(nodes, -1ul, 1u, 6u, 0u, triangle_return_mode, 0u);
    uint4 res = image_bvh_intersect_ray(nodeIdx, isect->t, P, dir, idir, bvh_desc);

    uint node_type = NODE_TYPE(nodeIdx);

    if (node_type == BoxNode32) {
      if (BOX_NODE_UPDATE(&traversal_stack[0], &stack_ptr, res, top_level, &nodeIdx))
        continue;
    }
    else {
      __global TriangleNode const *restrict tri = (__global TriangleNode const *restrict)(
          nodes + ((nodeIdx) >> 3u));

      if (node_type == Leaf_Node) {

        int prim_adr = tri->shape_id + prim_offset;
        uint type = tri->prim_type;
        bool hit = false;

#ifdef __VISIBILITY_FLAG__
        if (tri->prim_visibility & visibility)
#endif
        {
          switch (type & PRIMITIVE_ALL) {
            case PRIMITIVE_TRIANGLE: {

              hit = tri_intersect(as_float4(res), isect);

              break;
            }
#if BVH_FEATURE(BVH_MOTION)
            case PRIMITIVE_MOTION_TRIANGLE: {
              hit = motion_tri_intersect(kg, isect, tri, P, dir, ray->time, object, prim_adr);
              break;
            }
#endif

#if BVH_FEATURE(BVH_HAIR)
            case PRIMITIVE_CURVE_THICK:
            case PRIMITIVE_MOTION_CURVE_THICK:
            case PRIMITIVE_CURVE_RIBBON:
            case PRIMITIVE_MOTION_CURVE_RIBBON: {

              HairNode *hair = (HairNode *)(tri);

              hit = hair_intersect(kg, hair, isect, object, P, dir, ray->time, type);

              break;
            }
#endif
          }

          if (hit) {
            isect->prim = prim_adr;
            isect->object = object;
            isect->type = type;

            if (visibility & PATH_RAY_SHADOW_OPAQUE)
              return true;
          }
        }
      }

      else {  // Object_Node)

        object = tri->shape_id;

#if BVH_FEATURE(BVH_MOTION)
        isect->t = bvh_instance_motion_push(kg, object, ray, &P, &dir, &idir, isect->t, &ob_itfm);
#else

        ObjectNode *object_node = (ObjectNode *)(tri);
        Transform tfm = object_node->tfm;
        P = transform_point(&tfm, ray->P);

        float length;
        dir = bvh_clamp_direction(normalize_len(transform_direction(&tfm, ray->D), &length));
        idir = bvh_inverse_direction(dir);
        trnasform_factor = len(transform_direction(&tfm, ray->D));
        if (isect->t != FLT_MAX) {
          isect->t *= length;
        }
#endif

        uint2 offset = kernel_tex_fetch(__bvh_amd_offset, object);
        prim_offset = offset.y;

        nodes = (__global Node *)(&kernel_tex_fetch(__bvh_amd, offset.x));

        nodeIdx = BoxNode32;

        continue;
      }
    }

    stack_ptr -= LOCAL_SIZE;

    nodeIdx = traversal_stack[stack_ptr];
    if (!top_level && (nodeIdx & TOP_LEVEL)) {

#if BVH_FEATURE(BVH_MOTION)
      isect->t = bvh_instance_motion_pop(kg, object, ray, &P, &dir, &idir, isect->t, &ob_itfm);
#else
      if (isect->t != FLT_MAX)
        isect->t /= trnasform_factor;
#endif
      P = ray->P;
      dir = bvh_clamp_direction(ray->D);
      idir = bvh_inverse_direction(dir);
      object = OBJECT_NONE;
      prim_offset = 0;
      nodes = root;
    }
  }
  return (isect->prim != PRIM_NONE);
}

#undef BOX_NODE_UPDATE
