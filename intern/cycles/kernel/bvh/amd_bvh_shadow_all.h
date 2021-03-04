/*
 ?
 */
#define BOX_NODE_UPDATE box_node_update

// ccl_device_inline
inline __attribute__((always_inline)) bool BVH_FUNCTION_FULL_NAME(ABVH)(KernelGlobals *kg,
                                                                        const Ray *ray,
                                                                        Intersection *isect_array,
                                                                        const uint visibility,
                                                                        const uint max_hits,
                                                                        uint *num_hits)
{
#ifdef LOCAL_STACK
  __local uint *traversal_stack = &(kg->traversal_stack[get_local_id(0)]);
#else
  uint traversal_stack[STACK_SIZE];
#endif

  traversal_stack[0] = INVALID_NODE;

  int stack_ptr = LOCAL_SIZE;

  const float tmax = ray->t;
  float3 P = ray->P;
  float3 dir = bvh_clamp_direction(ray->D);
  float3 idir = bvh_inverse_direction(dir);
  int object = OBJECT_NONE;
  float isect_t = tmax;

#if BVH_FEATURE(BVH_MOTION)
  Transform ob_itfm;
#endif

  int num_hits_in_instance = 0;

  *num_hits = 0;
  isect_array->t = tmax;

  uint prim_offset = 0;

  uint triangle_return_mode = 1;

  uint nodeIdx = BoxNode32 | TOP_LEVEL;

  __global Node const *restrict root = (__global Node *)(&kernel_tex_fetch(__bvh_amd, 0));
  __global Node const *restrict nodes = root;

  while (nodeIdx != INVALID_NODE) {

    uint top_level = nodeIdx & TOP_LEVEL;

    nodeIdx &= (~TOP_LEVEL);

    ulong2 bvh_desc = pack_bvh_descriptor(nodes, -1ul, 1u, 6u, 0u, triangle_return_mode, 0u);
    uint4 res = image_bvh_intersect_ray(nodeIdx, isect_t, P, dir, idir, bvh_desc);

    if (NODE_TYPE(nodeIdx) == 0) {

      __global TriangleNode_ const *restrict tri = (__global TriangleNode_ const *restrict)(
          nodes + ((nodeIdx) >> 3u));

      int prim_adr = tri->shape_id + prim_offset;
      uint type = PRIMITIVE_TRIANGLE;
      //tri->prim_type;
      bool hit = false;
#ifdef __VISIBILITY_FLAG__
      if (kernel_tex_fetch(__prim_visibility, prim_adr) & visibility)
#endif
      {
        switch (type & PRIMITIVE_ALL) {
          case PRIMITIVE_TRIANGLE: {

            {

              hit = tri_intersect_(as_float4(res), isect_array);

              break;
            }
          }

#if BVH_FEATURE(BVH_MOTION)
          case PRIMITIVE_MOTION_TRIANGLE: {
            hit = motion_tri_intersect(kg, isect_array, tri, P, dir, ray->time, object, prim_adr);
            break;
          }
#endif

#if BVH_FEATURE(BVH_HAIR)
          case PRIMITIVE_CURVE_THICK:
          case PRIMITIVE_MOTION_CURVE_THICK:
          case PRIMITIVE_CURVE_RIBBON:
          case PRIMITIVE_MOTION_CURVE_RIBBON: {

            hit = hair_intersect(
                kg, (HairNode *)(tri), isect_array, object, P, dir, ray->time, type);
            break;
          }
#endif
        }
      }

      if (hit) {

        isect_array->prim = prim_adr;
        isect_array->type = type;
        isect_array->object = object;

        int prim = kernel_tex_fetch(__prim_index, isect_array->prim);
        // int prim = ((HairNode*)(tri))->prim_idx;
        int shader = 0;

#ifdef __HAIR__
        // if (kernel_tex_fetch(__prim_type, isect_array->prim) & PRIMITIVE_ALL_TRIANGLE)
        if (isect_array->type & PRIMITIVE_ALL_TRIANGLE)
#endif
        {
          shader = kernel_tex_fetch(__tri_shader, prim);
        }
#ifdef __HAIR__
        else {
          float4 str = kernel_tex_fetch(__curves, prim);
          shader = __float_as_int(str.z);
        }
#endif
        int flag = kernel_tex_fetch(__shaders, (shader & SHADER_MASK)).flags;

        if (!(flag & SD_HAS_TRANSPARENT_SHADOW)) {
          return true;
        }
        else if (*num_hits == max_hits) {
          return true;
        }

        isect_array++;
        (*num_hits)++;

        num_hits_in_instance++;

        isect_array->t = isect_t;
      }
    }

    else if (NODE_TYPE(nodeIdx) == Object_Node) {

      __global TriangleNode_ const *restrict tri = (global TriangleNode_ const *restrict)(
          nodes + ((nodeIdx >> 3u)));

      object = tri->shape_id;

#if BVH_FEATURE(BVH_MOTION)
      isect_t = bvh_instance_motion_push(kg, object, ray, &P, &dir, &idir, isect_t, &ob_itfm);
#else
      ObjectNode *object_node = (ObjectNode *)(tri);
      Transform tfm = object_node->tfm;
      P = transform_point(&tfm, ray->P);

      float len;
      dir = bvh_clamp_direction(normalize_len(transform_direction(&tfm, ray->D), &len));
      idir = bvh_inverse_direction(dir);

      if (isect_t != FLT_MAX)
        isect_t *= len;
#endif

      uint2 offset = kernel_tex_fetch(__bvh_amd_offset, object);
      prim_offset = offset.y;

      nodes = (__global Node *)(&kernel_tex_fetch(__bvh_amd, offset.x));

      nodeIdx = BoxNode32;
      num_hits_in_instance = 0;
      isect_array->t = isect_t;

      continue;
    }
    else {

      if (BOX_NODE_UPDATE(&traversal_stack[0], &stack_ptr, res, top_level, &nodeIdx))
        continue;
    }

    stack_ptr -= LOCAL_SIZE;

    nodeIdx = traversal_stack[stack_ptr];

    if (!top_level && (nodeIdx & TOP_LEVEL)) {
      if (num_hits_in_instance) {
        float t_fac;

#if BVH_FEATURE(BVH_MOTION)
        t_fac = 1.0f / len(transform_direction(&ob_itfm, ray->D));
#else
        Transform tfm = object_fetch_transform(kg, object, OBJECT_INVERSE_TRANSFORM);
        t_fac = 1.0f / len(transform_direction(&tfm, ray->D));
#endif
        for (int i = 0; i < num_hits_in_instance; i++) {
          (isect_array - i - 1)->t *= t_fac;
        }
      }

      P = ray->P;
      dir = bvh_clamp_direction(ray->D);
      idir = bvh_inverse_direction(dir);

      isect_t = tmax;
      isect_array->t = isect_t;
      object = OBJECT_NONE;
      prim_offset = 0;
      nodes = root;
    }
  }

  return false;
}

#undef BOX_NODE_UPDATE
