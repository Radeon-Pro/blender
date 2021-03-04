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

  struct ray_t __ray;
  __ray.origin = ray->P;
  __ray.direction = bvh_clamp_direction(ray->D);
  __ray.t_min = 0;
  __ray.t_max = ray->t;

  isect->t = ray->t;
  isect->u = 0.0f;
  isect->v = 0.0f;
  isect->prim = PRIM_NONE;
  isect->object = OBJECT_NONE;

  traversal_flags_t __flags = (visibility & PATH_RAY_SHADOW_OPAQUE) ?
                                  CL_AMD_TRAVERSAL_FLAGS_ACCEPT_FIRST_HIT :
                                  CL_AMD_TRAVERSAL_FLAGS_ACCEPT_CLOSEST_HIT;

  __global char *bvh_ptr = (__global char *)(&kernel_tex_fetch(__bvh_amd, 0));
  traversal_t __traversal = CreateTraversal_AMD(bvh_ptr, __ray, __flags);
  traversal_status_t __status = NextHit_AMD(&__traversal, &traversal_stack[0]);

  if (__status == CL_AMD_TRAVERSAL_STATUS_HIT) {
    isect->t = GetHitT_AMD(&__traversal);
    isect->prim = GetHitPrimitiveID_AMD(&__traversal);
    isect->object = GetHitInstanceID_AMD(&__traversal);
    // if (GetPrimitiveType(&__traversal) & PRIMITIVE_TRIANGLE) // can check against known/unknow
    // flag
    {
      float2 uv = GetHitBarycentrics_AMD(&__traversal);
      isect->u = uv.x;
      isect->v = uv.y;
      isect->type = PRIMITIVE_TRIANGLE;
    }
  }

  return (isect->prim != PRIM_NONE);
}
