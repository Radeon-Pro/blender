
ulong2 pack_bvh_descriptor(const __global Node* bvh_ptr,
                           ulong size,
                           uint box_sort_en,
                           uint box_grow_ulp,
                           uint big_page,
                           uint triangle_return_mode,
                           uint llc_noalloc)
{

    ulong base_address = ((const __global uchar *)bvh_ptr) - ((const global uchar *)0x0);
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

float3 inverse(float3 value)
{
    float3 inversed = { 1.0f / value.x, 1.0f / value.y, 1.0f / value.z };
    return inversed;
}

__attribute__((always_inline)) bool box_node_update(
#  ifdef LOCAL_STACK
    __local 
	#endif
	uint* traversal_stack,
    uint* ptr,
    uint4 res,
    uint top_level,
    uint* nodeIdx)
{
    if (res.w != INVALID_NODE | res.z != INVALID_NODE | res.y != INVALID_NODE |
        res.x != INVALID_NODE)
    {
        uint flag = top_level ? TOP_LEVEL : 0x0;
        int stack_ptr = *ptr;

        if (stack_ptr < STACK_SIZE && res.w != INVALID_NODE)
        {
            traversal_stack[stack_ptr] = res.w | flag;
            stack_ptr += LOCAL_SIZE;
        }

        if (stack_ptr < STACK_SIZE && res.z != INVALID_NODE)
        {
            traversal_stack[stack_ptr] = res.z |flag;
            stack_ptr += LOCAL_SIZE;
        }

        if (stack_ptr < STACK_SIZE && res.y != INVALID_NODE)
        {
            traversal_stack[stack_ptr] = res.y | flag;
            stack_ptr += LOCAL_SIZE;
        }

        *ptr = stack_ptr;
        *nodeIdx = (res.x | flag);
        return true;
    }

    return false;
}

bool tri_intersect(float4 res, struct traversal_t* traversal)
{
    float inv_denom = native_recip(res.y);
    float f = res.x * inv_denom;

    if (f < traversal->hit.t)
    {
        traversal->hit.uv.x = res.z * inv_denom;
        traversal->hit.uv.y = res.w * inv_denom;
        traversal->hit.t = f;
        return true;
    }
    return false;
}

float2 compute_uv(float3 v0, float3 v1, float3 v2, float3 direction, float3 origin)
{
    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    float3 s1 = cross(direction, e2);

    float  denom = dot(s1, e1);

    float  invd = 1.0f / denom;
    float3 d = origin - v0;
    float  b1 = dot(d, s1) * invd;

    float3 s2 = cross(d, e1);
    float  b2 = dot(direction, s2) * invd;

    float2 uv = { b1, b2 };
    return uv;
}
/*
float3 transform_point(const struct Transform* t, const float3 a)
{
    float3 c = (float3)(a.x * t->x.x + a.y * t->x.y + a.z * t->x.z + t->x.w,
        a.x * t->y.x + a.y * t->y.y + a.z * t->y.z + t->y.w,
        a.x * t->z.x + a.y * t->z.y + a.z * t->z.z + t->z.w);

    return c;
}

float3 transform_direction(const struct Transform* t, const float3 a)
{
    float3 c = (float3)(a.x * t->x.x + a.y * t->x.y + a.z * t->x.z,
        a.x * t->y.x + a.y * t->y.y + a.z * t->y.z,
        a.x * t->z.x + a.y * t->z.y + a.z * t->z.z);

    return c;
}*/

float3 normalize_length(const float3 a, float* t)
{
    *t = length(a);
    float x = 1.0f / *t;
    return a * x;
}

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
