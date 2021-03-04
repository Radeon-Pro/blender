#pragma once



#include "device/opencl/cl_amd_raytracing.h"
#include <vector>
#include <map>
#include "bvh/bvh_node_hwi.h"

class BVHEncoder
{
public:
    BVHEncoder(
        cl_command_queue command_queue,
        cl_amd_user_bvh_node_ptr root,
        const cl_amd_user_bvh_callbacks* callbacks,
        cl_mem out_buffer,
        cl_uint num_events_in_wait_list, const cl_event* event_wait_list,
        cl_event* event, bool in_top_level) :
        command_queue_(command_queue), root_(root), callbacks_(callbacks), out_buffer_(out_buffer), 
        num_events_in_wait_list_(num_events_in_wait_list),
       event_wait_list_(event_wait_list),
       event_(event),
       top_level_nodes_count_(0),
       is_top_level(in_top_level)
    {
        ;
    }
    BVHEncoder(
        cl_amd_user_bvh_node_ptr root,
        const cl_amd_user_bvh_callbacks* callbacks,
               char *out_array,
               bool in_top_level)
        : root_(root), callbacks_(callbacks), out_array_cpu_(out_array), is_top_level(in_top_level)
    {
        ;
    }
    cl_int encode();
    static size_t get_output_size(cl_amd_user_bvh_node_ptr root, const cl_amd_user_bvh_callbacks* callbacks);
    char *get_encoded_bvh()
    {
      return (&out_buffer_cpu_[0]);
    }
    size_t get_bvh_size()
    {
      return out_buffer_cpu_.size();
    }

private:
    uint encode_bvh(cl_amd_user_bvh_node_ptr current_root, std::vector<ABVHNode>& nodes, uint& abvh_cnt);
    uint flatten_nodes(cl_amd_user_bvh_node_ptr current_node, std::vector<ABVHNode>& nodes, uint depth, uint& abvh_cnt, cl_amd_aabb current_aabb);
    uint pack_box_node_f16(ABVHNode* nodes, bool valid_children[4], cl_amd_aabb bounds[4], uint& abvh_cnt);
    uint pack_box_node_f32(ABVHNode* nodes, bool valid_children[4], cl_amd_aabb bounds[4], bool is_root, uint& abvh_cnt);
    uint pack_leaf(cl_amd_user_bvh_node_ptr current_node, std::vector<ABVHNode>& nodes, uint& abvh_cnt);

    void print_built_bvh(FILE* file, ABVHNode* root, ABVHNode* node, uint ptr);

    static int subtree_size(BVH_STAT stat, cl_amd_user_bvh_node_ptr node, const cl_amd_user_bvh_callbacks* callbacks);
    cl_int copy_to_device(size_t size);

    std::vector<char> out_buffer_cpu_;
    char* out_array_cpu_;

    cl_amd_user_bvh_node_ptr root_;
    const cl_amd_user_bvh_callbacks* callbacks_;
    cl_command_queue command_queue_;
    cl_mem out_buffer_;
    cl_uint num_events_in_wait_list_;
    const cl_event* event_wait_list_;
    cl_event* event_;
    bool is_top_level;

    std::vector<std::pair<cl_amd_user_bvh_node_ptr, ABVHNode*>> top_level_map;
    uint top_level_nodes_count_;

    enum node_type { triangle0 = 0, triangle1 = 1, triangle2 = 2, triangle3 = 3,
                     box16 = 4, box32 = 5, object = 6, custom = 7 };

};
