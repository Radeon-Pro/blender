#include "common_structures.h"
#include "common_functions.h"

struct traversal_t CreateTraversal_AMD(__global char* bvh_ptr, struct ray_t ray, traversal_flags_t flags)
{
    struct traversal_t traversal;
    traversal.node_ptr = bvh_ptr;
    traversal.ray = ray;
    traversal.flag = flags;
    return traversal;
}
__attribute__((always_inline))
traversal_status_t NextHit_AMD(struct traversal_t* traversal, 
#  ifdef LOCAL_STACK
__local 
#endif
uint* traversal_stack)
{
    traversal_stack[0] = INVALID_NODE;

    uint stack_ptr = LOCAL_SIZE;

    float3 origin = traversal->ray.origin;
    float3 dir = traversal->ray.direction;
    float3 idir = inverse(dir);
    int object = OBJECT_NONE;

    traversal->hit.t = traversal->ray.t_max;
    traversal->hit.uv.x = 0.0f;
    traversal->hit.uv.y = 0.0f;
    traversal->hit.primitive_id = PRIM_NONE;
    traversal->hit.instance_id = OBJECT_NONE;

    uint nodeIdx = BOX32 | TOP_LEVEL;
	
	uint prim_offset = 0;

    float tfm_factor;
    uint triangle_return_mode = 1;
    traversal->hit.status = CL_AMD_TRAVERSAL_STATUS_FINISHED;
    __global struct Node const* restrict nodes = (__global struct Node*)(traversal->node_ptr);
    __global struct Node const* restrict root = nodes;
	
	BoxNodeF32* box_node = (BoxNodeF32*)(traversal->node_ptr);
	uint root_offset = box_node->padding0;
	
    while (nodeIdx != INVALID_NODE)
    {
        uint top_level = nodeIdx & TOP_LEVEL;
        nodeIdx &= (~TOP_LEVEL);

        ulong2 bvh_desc = pack_bvh_descriptor(nodes, -1ul, 1u, 6u, 0u, triangle_return_mode, 0u);
        uint4 res = image_bvh_intersect_ray(nodeIdx, traversal->hit.t, origin,
                                                          dir, idir, bvh_desc);

        uint node_type = NODE_TYPE(nodeIdx);

        if (node_type == BOX32 || node_type == BOX16)
        {
            if (box_node_update(&traversal_stack[0], &stack_ptr, res, top_level, &nodeIdx))
            {
                continue;
            }
        }
        else
        {
            __global struct TriangleNode const* restrict tri = (__global struct TriangleNode const* restrict)(
                nodes + GET_NODE(nodeIdx));

            if (node_type == TRIANGLE0 || node_type == TRIANGLE1 || node_type == TRIANGLE2
                   || node_type == TRIANGLE3)
            {
                bool hit = tri_intersect(as_float4(res), traversal);
                if (hit)
                {
                     traversal->hit.primitive_id = tri->triangle_id + prim_offset;
                     traversal->hit.instance_id = object;
                     //traversal->node_ptr = (__global char*)(nodes + GET_NODE(nodeIdx));
                     traversal->hit.status = CL_AMD_TRAVERSAL_STATUS_HIT;
                     if (traversal->flag == CL_AMD_TRAVERSAL_FLAGS_ACCEPT_FIRST_HIT)
                     {
						 return traversal->hit.status;
                     }
                }
            }
            else if(node_type == OBJECT)
            {
				__global ObjectNode const* restrict object_node = (__global ObjectNode*)(nodes + GET_NODE(nodeIdx));
                Transform tfm = object_node->tfm;
				object = object_node->shape_id;
                origin = transform_point(&tfm, traversal->ray.origin);
				prim_offset = object_node->prim_offset;
                float len;
                dir = normalize_length(transform_direction(&tfm, traversal->ray.direction), &len);
                idir = inverse(dir);
                tfm_factor = length(transform_direction(&tfm, traversal->ray.direction));
                if (traversal->hit.t != FLT_MAX)
                {
                    traversal->hit.t *= len;
                }
				
                nodes = (__global struct Node*)(traversal->node_ptr + object_node->bottom_pointer + root_offset);
                nodeIdx = BOX32;
                continue;

            }
            else if (node_type == USER)
            {
                traversal->node_ptr = (__global char*)(nodes + GET_NODE(nodeIdx));
                traversal->hit.status = CL_AMD_TRAVERSAL_STATUS_UNKNOWN;
                if (traversal->flag == CL_AMD_TRAVERSAL_FLAGS_ACCEPT_FIRST_HIT)
                {
                    break;
                }
            }
        }

        stack_ptr -= LOCAL_SIZE;

        nodeIdx = traversal_stack[stack_ptr];

        if (!top_level && (nodeIdx & TOP_LEVEL))
        {
            if (traversal->hit.t != FLT_MAX)
            {
                traversal->hit.t /= tfm_factor;
            }

            origin = traversal->ray.origin;
            dir = traversal->ray.direction;
            idir = inverse(dir);
            object = OBJECT_NONE;
            nodes = root;
			prim_offset = 0;
			//traversal->node_ptr = (__global char*)(nodes);
        }

    }
    return traversal->hit.status;
}

__attribute__((always_inline))
uint GetHitInstanceID_AMD(const struct traversal_t* traversal)
{
    return traversal->hit.instance_id;
}
__attribute__((always_inline))
uint GetHitPrimitiveID_AMD(const struct traversal_t* traversal)
{
    return traversal->hit.primitive_id;
}
__attribute__((always_inline))
float GetHitT_AMD(const struct traversal_t* traversal)
{
    return traversal->hit.t;
}
__attribute__((always_inline))
float2 GetHitBarycentrics_AMD(const struct traversal_t* traversal)
{
    return traversal->hit.uv;
}
__attribute__((always_inline))
void* GetUserData_AMD(const struct traversal_t* traversal)
{
    CustomNode node = ((__global struct CustomNode const* restrict)traversal->node_ptr)[0];
    return (void*)node.data;
}
__attribute__((always_inline))
uint GetUserDataSize_AMD(const struct traversal_t* traversal)
{
    return ((__global struct CustomNode const* restrict)traversal->node_ptr)->size;
}
