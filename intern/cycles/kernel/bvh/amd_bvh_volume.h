/*

 */

#define BOX_NODE_UPDATE box_node_update

// ccl_device_inline
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

#if BVH_FEATURE(BVH_MOTION)
  Transform ob_itfm;
#endif

  isect->t = ray->t;
  isect->u = 0.0f;
  isect->v = 0.0f;
  isect->prim = PRIM_NONE;
  isect->object = OBJECT_NONE;

  uint prim_offset = 0;
  uint transform_offset = 0;

  uint triangle_return_mode = 1;

  uint nodeIdx = BoxNode32 | TOP_LEVEL;

  uint obj_volume = 0;

  __global Node const *restrict root = (__global Node *)(&kernel_tex_fetch(__bvh_amd, 0));
  __global Node const *restrict nodes = root;
  
  uint root_offset = ((__global BoxNodeF32*)(nodes))->padding1;

  while (nodeIdx != INVALID_NODE) {

    uint top_level = nodeIdx & TOP_LEVEL;

    nodeIdx &= (~TOP_LEVEL);

    ulong2 bvh_desc = pack_bvh_descriptor(nodes, -1ul, 1u, 6u, 0u, triangle_return_mode, 0u);
    uint4 res = image_bvh_intersect_ray(nodeIdx, isect->t, P, dir, idir, bvh_desc);

    if (NODE_TYPE(nodeIdx) == Leaf_Node) {

      __global TriangleNode const *restrict tri = (__global TriangleNode const *restrict)(
          nodes + ((nodeIdx) >> 3u));

      int prim_adr = tri->shape_id + prim_offset;
      uint type = tri->prim_type;
	  if(top_level)
	  object = tri->data1;
#ifdef __VISIBILITY_FLAG__
      if (tri->prim_visibility & visibility)
#endif
      {

        switch (type & PRIMITIVE_ALL) {
          case PRIMITIVE_TRIANGLE: {
			uint tri_object = (object == OBJECT_NONE) ? tri->data0 : object;
            int object_flag = kernel_tex_fetch(__object_flag, tri_object);
            if ((object_flag & SD_OBJECT_HAS_VOLUME) != 0) {

              if (tri_intersect(as_float4(res), isect)) {
                isect->prim = prim_adr;
                isect->type = PRIMITIVE_TRIANGLE;
                isect->object = object;
              }
            }

            break;
          }

#if BVH_FEATURE(BVH_MOTION)
          case PRIMITIVE_MOTION_TRIANGLE: {
            uint tri_object = (object == OBJECT_NONE) ? kernel_tex_fetch(__prim_object, prim_adr) :
                                                        object;
            int object_flag = kernel_tex_fetch(__object_flag, tri_object);
            if ((object_flag & SD_OBJECT_HAS_VOLUME) != 0) {
              if (motion_tri_intersect(kg, isect, tri, P, dir, ray->time, object, prim_adr)) {
                isect->prim = prim_adr;
                isect->type = PRIMITIVE_MOTION_TRIANGLE;
                isect->object = object;
              }
            }
            break;
          }
#endif
        }
      }
    }

    else if (NODE_TYPE(nodeIdx) == Object_Node) {

      __global TriangleNode const *restrict tri = (global TriangleNode const *restrict)(
          nodes + ((nodeIdx >> 3u)));

      object = tri->shape_id;
      int object_flag = kernel_tex_fetch(__object_flag, object);
      if (object_flag & SD_OBJECT_HAS_VOLUME) {
		  ObjectNode *object_node = (ObjectNode *)(tri);
#if BVH_FEATURE(BVH_MOTION)
        isect->t = bvh_instance_motion_push(kg, object, ray, &P, &dir, &idir, isect->t, &ob_itfm);
#else

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

        //uint4 offset = kernel_tex_fetch(__bvh_amd_offset, object);
        //prim_offset = offset.y;
		//transform_offset = offset.z;
		uint offset = object_node->data_used[1];
        nodes = (__global Node *)(&kernel_tex_fetch(__bvh_amd, offset + root_offset));
		BoxNodeF32 box_node_ = *(__global BoxNodeF32*)(nodes);
		prim_offset = box_node_.prim_offset;

        nodeIdx = BoxNode32;
      }
      else {

        stack_ptr -= LOCAL_SIZE;
        nodeIdx = traversal_stack[stack_ptr];
        object = OBJECT_NONE;
      }
      continue;
    }
    else {
		__global BoxNodeF32* box_node = (__global BoxNodeF32*)(nodes + (nodeIdx >> 3u));
		if (box_node->padding3 & visibility)
		{
		#if BVH_FEATURE(BVH_HAIR)

		if (box_node->padding2 > 0 && (box_node->padding3 & visibility))
		{
			/*uint transform_id = 4*box_node->padding0 + transform_offset;
			res = unaligned_box_intersect(kg, box_node, transform_id, dir, P, isect->t);*/
			__global Unaligned_BoxNode* unaligned_node = (__global Unaligned_BoxNode*)(nodes + (nodeIdx >> 3u));			
			res = unaligned_box_intersect(unaligned_node, dir, P, isect->t);
		}
		#endif

      if (BOX_NODE_UPDATE(&traversal_stack[0], &stack_ptr, res, top_level, &nodeIdx))
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
