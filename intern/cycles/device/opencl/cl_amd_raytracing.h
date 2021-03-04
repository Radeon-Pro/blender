#ifndef CL_AMD_RAYTRACING_H
#define CL_AMD_RAYTRACING_H

#include "../../extern/clew/include/clew.h"
/** DRAFT RAYTRACING OPENCL EXTENSION **/
// Invalid node pointer.
#define CL_AMD_INVALID_NODE_PTR 0xffffffff

// Node type
#define CL_AMD_BVH_NODE_TYPE_AABB 1
#define CL_AMD_BVH_NODE_TYPE_INSTANCE 2
#define CL_AMD_BVH_NODE_TYPE_TRINAGLE 3
#define CL_AMD_BVH_NODE_TYPE_USER 4

// Info
#define CL_AMD_EXTERNAL_BVH_SIZE 1
#define CL_AMD_BOTTOM_LEVEL_BVH_SIZE 2 
#define CL_AMD_TOP_LEVEL_BVH_SIZE 3 
#define CL_AMD_BOTTOM_LEVEL_SCRATCH_BUFFER_SIZE 4 
#define CL_AMD_TOP_LEVEL_SCRATCH_BUFFER_SIZE 5

//  cl_amd_bottom_level_build_flags
#define CL_AMD_BUILD_PREFER_FAST_BUILD 1
#define CL_AMD_BUILD_PREFER_FAST_TRACE 2

// cl_amd_index_format
#define CL_AMD_INDEX_FORMAT_UINT16 1
#define CL_AMD_INDEX_FORMAT_UINT32 2

// User provided node ptr for a user BVH to HW BVH conversion.
typedef void* cl_amd_user_bvh_node_ptr;
typedef cl_uint cl_amd_bvh_node_type;
typedef cl_uint cl_amd_bottom_level_build_flags;
typedef cl_uint cl_amd_top_level_build_flags;
typedef cl_uint cl_amd_index_format;
typedef cl_uint cl_amd_external_bvh_info;
typedef cl_uint cl_amd_bottom_level_info;
typedef cl_uint cl_amd_top_level_info;

/** Axis-aligned bounding box **/
typedef struct _cl_amd_aabb
{
    cl_float3 pmin;
    cl_float3 pmax;
} cl_amd_aabb;

/** Triangle definition **/
typedef struct _cl_amd_triangle
{
    cl_float3 v0;
    cl_float3 v1;
    cl_float3 v2;
    cl_uint id;
} cl_amd_triangle;

/** Transformation matrix 3x4 **/
typedef struct _cl_amd_transform
{
    float m[12];
} cl_amd_transform;

/** BVH traversal callback interface **/
typedef struct _cl_amd_user_bvh_callbacks 
{
    // Get the type of a given node.
    cl_amd_bvh_node_type (*pfn_get_node_type)(cl_amd_user_bvh_node_ptr ptr);
    // IF TYPE == BVH_NODE_TYPE_AABB
    // Get the number of chilren nodes of a given node.
    cl_uint (*pfn_get_num_children)(cl_amd_user_bvh_node_ptr ptr);
    // Get i-th child node of a given node.
    cl_amd_user_bvh_node_ptr (*pfn_get_child)(cl_amd_user_bvh_node_ptr ptr, cl_uint i);   
    // Get i-th child bounding box of a given node. 
    cl_amd_aabb (*pfn_get_child_aabb)(cl_amd_user_bvh_node_ptr ptr, cl_uint i);
    // IF TYPE == BVH_NODE_TYPE_TRINAGLE
    // Get triangle count in a given node.
    cl_uint (*pfn_get_num_triangles)(cl_amd_user_bvh_node_ptr ptr);
    // Get i-th triangle in a given node.
    cl_amd_triangle (*pfn_get_triangle)(cl_amd_user_bvh_node_ptr ptr, cl_uint i);
    cl_int (*pfn_get_primitive_id)(cl_amd_user_bvh_node_ptr ptr);
    // IF TYPE == BVH_NODE_TYPE_INSTANCE.
    // Get BLAS pointer for a given node.
    cl_amd_user_bvh_node_ptr (*pfn_get_blas)(cl_amd_user_bvh_node_ptr ptr);
    // Get transform for a given node.
    cl_amd_transform (*pfn_get_transform)(cl_amd_user_bvh_node_ptr ptr);
    cl_int (*pfn_get_instance_id)(cl_amd_user_bvh_node_ptr ptr);
    cl_uint (*pfn_get_blas_offset)(cl_amd_user_bvh_node_ptr ptr);
    cl_uint (*pfn_get_blas_prim_offset)(cl_amd_user_bvh_node_ptr ptr);
    // IF TYPE == BVH_NODE_TYPE_CUSTOM
    // Get the size of the custom data chunk (up to 128 bytes)
    cl_uint (*pfn_get_user_data_size)(cl_amd_user_bvh_node_ptr ptr);
    // Custom data.
    void* (*pfn_get_user_data)(cl_amd_user_bvh_node_ptr ptr);
} cl_amd_user_bvh_callbacks;

/** Bottom level structure build input **/
typedef struct _cl_amd_bottom_level_bvh_build_input
{
    cl_mem vertex_buffer;
    size_t vertex_count;
    size_t vertex_offset;
    cl_uint vertex_stride;
    cl_mem index_buffer;
    size_t index_count;
    size_t index_offset;
    cl_uint index_stride;
    cl_amd_index_format format;
} cl_amd_bottom_level_bvh_build_input;

/**
 * @brief Get an info about external BVH.
 *
 * @param root Pointer to the root BVH node.
 * @param callbacks BVH iterator interface.
 * @param param_name The name of the requested parameter.
 * @param param_value_size The size of the parameter value.
 * @param param_value The location to write a value to.
 * @param param_value_size_ret The actual number of bytes written.
 *
 * @return Error code.
 * **/
cl_int clGetExternalBVHInfo_AMD(cl_amd_user_bvh_node_ptr root, const cl_amd_user_bvh_callbacks* callbacks,
                                   cl_amd_external_bvh_info param_name, size_t param_value_size, void* param_value, 
                                   size_t* param_value_size_ret);

/**
 * @brief Encode external BVH into hardware format.
 *
 * This function expects application generated two level BVH in host memory pointed by root parameter.
 * The function makes use of the provided callbacks structure to iterate over the BVH. At every node starting from 
 * the root, the function queries node parameters via provided cl_amd_user_bvh_callbacks structure and encodes the 
 * node into an internal hardware format. The function is executed on CPU and command_queue parameter is only used 
 * to map and fill the buffer.
 *
 * @param command_queue OpenCL to map bvh_buffer on.
 * @param root Pointer to the root BVH node.
 * @param callbacks BVH iterator interface.
 * @param bvh_buffer Resulting BVH buffer.
 * @param num_events_in_wait_list Number of events to wait for.
 * @param event_wait_list Events to wait for.
 * @param event_wait_list Event associated with this call.
 *
 * @return Error code.
 * **/
cl_int clEnqueueEncodeExternalBVH_AMD(cl_command_queue command_queue, cl_amd_user_bvh_node_ptr root, 
                                     const cl_amd_user_bvh_callbacks* callbacks, cl_mem bvh_buffer,
                                     cl_uint num_events_in_wait_list, const cl_event* event_wait_list, 
                                     cl_event* event);

/**
 * @brief Encode external BVH into hardware format.
 *
 * This function expects application generated two level BVH in host memory pointed by root parameter.
 * The function makes use of the provided callbacks structure to iterate over the BVH. At every node starting from
 * the root, the function queries node parameters via provided cl_amd_user_bvh_callbacks structure and encodes the
 * node into an internal hardware format. Resulting BVH is written in dynamic char array provided by user and
 * can be later transfered to GPu on application side.
 *
 * @param root Pointer to the root BVH node.
 * @param callbacks BVH iterator interface.
 * @param bvh_buffer Resulting BVH array (must be pre-allocated on user side).
 *
 * @return Error code.
 * **/
cl_int clEnqueueEncodeExternalBVH_AMD(cl_amd_user_bvh_node_ptr root,
    const cl_amd_user_bvh_callbacks* callbacks, char* bvh_buffer);

/** 
 * @brief Get an info about a bottom level BVH.
 *
 * @param num_build_inputs The number of build inputs in build_inputs.
 * @param build_inputs An array of num_build_inputs structures.
 *
 * @return Error code.
 * **/
cl_int clGetBottomLevelBVHInfo_AMD(cl_uint num_build_inputs, const cl_amd_bottom_level_bvh_build_input* build_inputs,
                                      cl_amd_bottom_level_info param_name, size_t param_value_size, void* param_value, 
                                      size_t* param_value_size_ret);
/** 
 * @brief Get an info about a top level BVH.
 *
 * @param instance_count The number of build inputs in instance_buffer.
 *
 * @return Error code.
 * **/
cl_int clGetTopLevelBVHInfo_AMD(cl_uint instance_count, cl_amd_top_level_info param_name,
                                   size_t param_value_size, void* param_value, size_t* param_value_size_ret);

/** 
 * @brief Enqueue bottom level BVH build.
 *
 * @param num_build_inputs The number of build inputs in build_inputs.
 * @param build_inputs An array of num_build_inputs build input structures.
 * @param callbacks BVH iterator interface.
 * @param bvh_buffer Resulting BVH buffer.
 * @param num_events_in_wait_list Number of events to wait for.
 * @param event_wait_list Events to wait for.
 * @param event_wait_list Event associated with this call.
 *
 * @return Error code.
 * **/
cl_int clEnqueueBuildBottomLevelBVH_AMD(cl_command_queue command_queue, cl_uint num_build_inputs,
                                        const cl_amd_bottom_level_bvh_build_input* build_inputs,
                                        cl_amd_bottom_level_build_flags build_flags, cl_mem scratch_buffer,
                                        cl_mem bvh_buffer, cl_uint num_events_in_wait_list,
                                        const cl_event* event_wait_list, cl_event* event);

/** 
 * @brief Enqueue top level BVH build.
 *
 * @param instance_count The number of instance structures in instance_buffer.
 * @param build_inputs An array of num_build_inputs build input structures.
 * @param callbacks BVH iterator interface.
 * @param bvh_buffer Resulting BVH buffer.
 * @param num_events_in_wait_list Number of events to wait for.
 * @param event_wait_list Events to wait for.
 * @param event_wait_list Event associated with this call.
 *
 * @return Error code.
 * **/
cl_int clEnqueueBuildTopLevelBVH_AMD(cl_command_queue command_queue, cl_uint instance_count, cl_mem instance_buffer,
                                     cl_amd_top_level_build_flags build_flags, cl_mem scratch_buffer,
                                     cl_mem bvh_buffer, cl_uint num_events_in_wait_list, 
                                     const cl_event* event_wait_list, cl_event* event);

#endif
