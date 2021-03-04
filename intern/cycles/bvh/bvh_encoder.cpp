#include "bvh_encoder.h"
#include <cassert>
#include "util/util_half_float.h"

#define FULL_PRECISION_DEPTH 4

#define MAKE_BOX_MIN(n, s, t) box_f##t.box##n##_min_##s
#define MAKE_BOX_MAX(n, s, t) box_f##t.box##n##_max_##s
#define CAST_HALF_DOWN(x) half_float::detail::float2half<std::round_toward_neg_infinity>(x)
#define CAST_HALF_UP(x) half_float::detail::float2half<std::round_toward_infinity>(x)

size_t align_up(size_t offset, size_t alignment)
{
    return (offset + alignment - 1) & ~(alignment - 1);
}

float dot(float x[3], float y[3])
{
    return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
}

cl_amd_aabb unite_aabb(cl_amd_aabb box1, cl_amd_aabb box2)
{
    cl_amd_aabb united;
    united.pmax.x = std::max(box1.pmax.x, box2.pmax.x);
    united.pmax.y = std::max(box1.pmax.y, box2.pmax.y);
    united.pmax.z = std::max(box1.pmax.z, box2.pmax.z);

    united.pmin.x = std::min(box1.pmin.x, box2.pmin.x);
    united.pmin.y = std::min(box1.pmin.y, box2.pmin.y);
    united.pmin.z = std::min(box1.pmin.z, box2.pmin.z);

    return united;
}

cl_amd_transform transform_quick_inverse(cl_amd_transform M)
{
    cl_amd_transform R;
    float det = M.m[0] * (M.m[8] * M.m[4] - M.m[5] * M.m[7]) - M.m[1] * (M.m[8] * M.m[3] - M.m[5] * M.m[6]) +
        M.m[2] * (M.m[7] * M.m[3] - M.m[4] * M.m[6]);
    if (det == 0.0f)
    {
        M.m[0] += 1e-8f;
        M.m[4] += 1e-8f;
        M.m[8] += 1e-8f;
        det = M.m[0] * (M.m[8] * M.m[4] - M.m[5] * M.m[7]) - M.m[1] * (M.m[8] * M.m[3] - M.m[5] * M.m[6]) +
            M.m[2] * (M.m[7] * M.m[3] - M.m[4] * M.m[6]);
    }
    det = (det != 0.0f) ? 1.0f / det : 0.0f;

    float Rx[3];
    Rx[0] = det * (M.m[8] * M.m[4] - M.m[5] * M.m[7]);
    Rx[1] = det * (M.m[5] * M.m[6] - M.m[8] * M.m[3]);
    Rx[2] = det * (M.m[7] * M.m[3] - M.m[4] * M.m[6]);

    float Ry[3];
    Ry[0] = det * (M.m[2] * M.m[7] - M.m[8] * M.m[1]);
    Ry[1] = det * (M.m[8] * M.m[0] - M.m[2] * M.m[6]);
    Ry[2] = det * (M.m[1] * M.m[6] - M.m[7] * M.m[0]);

    float Rz[3];
    Rz[0] = det * (M.m[7] * M.m[1] - M.m[2] * M.m[4]);
    Rz[1] = det * (M.m[2] * M.m[3] - M.m[5] * M.m[0]);
    Rz[2] = det * (M.m[4] * M.m[0] - M.m[1] * M.m[3]);

    float T[3];
    T[0] = -M.m[9], T[1] = -M.m[10], T[2] -M.m[11];

    R.m[0] = Rx[0], R.m[1] = Rx[1], R.m[2]  = Rx[2], R.m[3]  = dot(Rx, T);
    R.m[4] = Ry[0], R.m[5] = Ry[1], R.m[6]  = Ry[2], R.m[7]  = dot(Ry, T);
    R.m[8] = Rz[0], R.m[9] = Rz[1], R.m[10] = Rz[2], R.m[11] = dot(Rz, T);

    return R;
}

size_t BVHEncoder::get_output_size(cl_amd_user_bvh_node_ptr root, const cl_amd_user_bvh_callbacks* callbacks)
{
    size_t num_nodes = subtree_size(BVH_STAT_NODE_COUNT, root, callbacks);
    size_t num_leaf_nodes = subtree_size(BVH_STAT_LEAF_COUNT, root, callbacks);
    assert(num_leaf_nodes <= num_nodes);
    size_t num_inner_nodes = num_nodes - num_leaf_nodes;

    int tentative_size = (num_leaf_nodes + 100) * 2;
    size_t amd_node_size = tentative_size * sizeof(ABVHNode);
    size_t hw_alighned = align_up(amd_node_size, PADDING_SIZE);

    return hw_alighned;
}

int BVHEncoder::subtree_size(BVH_STAT stat, cl_amd_user_bvh_node_ptr node, const cl_amd_user_bvh_callbacks* callbacks)
{
    int count = 0;
    bool is_leaf = (callbacks->pfn_get_node_type(node) != CL_AMD_BVH_NODE_TYPE_AABB);

    switch (stat)
    {
    case BVH_STAT_NODE_COUNT:
        count = 1;
        break;
    case BVH_STAT_LEAF_COUNT:
        count = is_leaf ? 1 : 0;
        break;
    }

    if (!is_leaf)
    {
        uint32_t num_children = callbacks->pfn_get_num_children(node);
        for (int i = 0; i < num_children; ++i)
        {
            cl_amd_user_bvh_node_ptr child_node = callbacks->pfn_get_child(node, i);
            count += subtree_size(stat, child_node, callbacks);
        }
    }

    return count;
}

uint BVHEncoder::pack_box_node_f16(ABVHNode* nodes, bool valid_children[4], cl_amd_aabb bounds[4], uint& abvh_cnt)
{
    BoxNodeF16 box_f16;

    box_f16.box0_max_x = box_f16.box0_max_y = box_f16.box0_max_z = -INFINITY;
    box_f16.box0_min_x = box_f16.box0_min_y = box_f16.box0_min_z = INFINITY;
    box_f16.box1_max_x = box_f16.box1_max_y = box_f16.box1_max_z = -INFINITY;
    box_f16.box1_min_x = box_f16.box1_min_y = box_f16.box1_min_z = INFINITY;
    box_f16.box2_max_x = box_f16.box2_max_y = box_f16.box2_max_z = -INFINITY;
    box_f16.box2_min_x = box_f16.box2_min_y = box_f16.box2_min_z = INFINITY;
    box_f16.box3_max_x = box_f16.box3_max_y = box_f16.box3_max_z = -INFINITY;
    box_f16.box3_min_x = box_f16.box3_min_y = box_f16.box3_min_z = INFINITY;

    if (valid_children[0])
    {
        cl_amd_aabb box = bounds[0];

        MAKE_BOX_MIN(0, x, 16) = CAST_HALF_DOWN(box.pmin.x);
        MAKE_BOX_MIN(0, y, 16) = CAST_HALF_DOWN(box.pmin.y);
        MAKE_BOX_MIN(0, z, 16) = CAST_HALF_DOWN(box.pmin.z);

        MAKE_BOX_MAX(0, x, 16) = CAST_HALF_UP(box.pmax.x);
        MAKE_BOX_MAX(0, y, 16) = CAST_HALF_UP(box.pmax.y);
        MAKE_BOX_MAX(0, z, 16) = CAST_HALF_UP(box.pmax.z);
    }

    if (valid_children[1])
    {
        cl_amd_aabb box = bounds[1];

        MAKE_BOX_MIN(1, x, 16) = CAST_HALF_DOWN(box.pmin.x);
        MAKE_BOX_MIN(1, y, 16) = CAST_HALF_DOWN(box.pmin.y);
        MAKE_BOX_MIN(1, z, 16) = CAST_HALF_DOWN(box.pmin.z);

        MAKE_BOX_MAX(1, x, 16) = CAST_HALF_UP(box.pmax.x);
        MAKE_BOX_MAX(1, y, 16) = CAST_HALF_UP(box.pmax.y);
        MAKE_BOX_MAX(1, z, 16) = CAST_HALF_UP(box.pmax.z);
    }

    if (valid_children[2])
    {
        cl_amd_aabb box = bounds[2];

        MAKE_BOX_MIN(2, x, 16) = CAST_HALF_DOWN(box.pmin.x);
        MAKE_BOX_MIN(2, y, 16) = CAST_HALF_DOWN(box.pmin.y);
        MAKE_BOX_MIN(2, z, 16) = CAST_HALF_DOWN(box.pmin.z);

        MAKE_BOX_MAX(2, x, 16) = CAST_HALF_UP(box.pmax.x);
        MAKE_BOX_MAX(2, y, 16) = CAST_HALF_UP(box.pmax.y);
        MAKE_BOX_MAX(2, z, 16) = CAST_HALF_UP(box.pmax.z);
    }

    if (valid_children[3])
    {
        cl_amd_aabb box = bounds[3];

        MAKE_BOX_MIN(3, x, 16) = CAST_HALF_DOWN(box.pmin.x);
        MAKE_BOX_MIN(3, y, 16) = CAST_HALF_DOWN(box.pmin.y);
        MAKE_BOX_MIN(3, z, 16) = CAST_HALF_DOWN(box.pmin.z);

        MAKE_BOX_MAX(3, x, 16) = CAST_HALF_UP(box.pmax.x);
        MAKE_BOX_MAX(3, y, 16) = CAST_HALF_UP(box.pmax.y);
        MAKE_BOX_MAX(3, z, 16) = CAST_HALF_UP(box.pmax.z);
    }

    int node_id_ptr = MAKE_PTR(abvh_cnt, node_type::box16);
    abvh_cnt++;

    ABVHNode& no = nodes[DECODE_NODE(node_id_ptr)];
    no.box = box_f16;
    return node_id_ptr;
}

uint BVHEncoder::pack_box_node_f32(ABVHNode* nodes, bool valid_children[4], cl_amd_aabb bounds[4], bool is_root, uint& abvh_cnt)
{
    BoxNodeF32 box_f32;

    box_f32.box0_max_x = box_f32.box0_max_y = box_f32.box0_max_z = -INFINITY;
    box_f32.box0_min_x = box_f32.box0_min_y = box_f32.box0_min_z = INFINITY;
    box_f32.box1_max_x = box_f32.box1_max_y = box_f32.box1_max_z = -INFINITY;
    box_f32.box1_min_x = box_f32.box1_min_y = box_f32.box1_min_z = INFINITY;
    box_f32.box2_max_x = box_f32.box2_max_y = box_f32.box2_max_z = -INFINITY;
    box_f32.box2_min_x = box_f32.box2_min_y = box_f32.box2_min_z = INFINITY;
    box_f32.box3_max_x = box_f32.box3_max_y = box_f32.box3_max_z = -INFINITY;
    box_f32.box3_min_x = box_f32.box3_min_y = box_f32.box3_min_z = INFINITY;

    if (valid_children[0])
    {
        cl_amd_aabb box = bounds[0];

        MAKE_BOX_MIN(0, x, 32) = (box.pmin.x);
        MAKE_BOX_MIN(0, y, 32) = (box.pmin.y);
        MAKE_BOX_MIN(0, z, 32) = (box.pmin.z);

        MAKE_BOX_MAX(0, x, 32) = (box.pmax.x);
        MAKE_BOX_MAX(0, y, 32) = (box.pmax.y);
        MAKE_BOX_MAX(0, z, 32) = (box.pmax.z);
    }
    if (valid_children[1])
    {
        cl_amd_aabb box = bounds[1];

        MAKE_BOX_MIN(1, x, 32) = (box.pmin.x);
        MAKE_BOX_MIN(1, y, 32) = (box.pmin.y);
        MAKE_BOX_MIN(1, z, 32) = (box.pmin.z);

        MAKE_BOX_MAX(1, x, 32) = (box.pmax.x);
        MAKE_BOX_MAX(1, y, 32) = (box.pmax.y);
        MAKE_BOX_MAX(1, z, 32) = (box.pmax.z);
    }
    if (valid_children[2])
    {
        cl_amd_aabb box = bounds[2];

        MAKE_BOX_MIN(2, x, 32) = (box.pmin.x);
        MAKE_BOX_MIN(2, y, 32) = (box.pmin.y);
        MAKE_BOX_MIN(2, z, 32) = (box.pmin.z);

        MAKE_BOX_MAX(2, x, 32) = (box.pmax.x);
        MAKE_BOX_MAX(2, y, 32) = (box.pmax.y);
        MAKE_BOX_MAX(2, z, 32) = (box.pmax.z);
    }
    if (valid_children[3])
    {
        cl_amd_aabb box = bounds[3];

        MAKE_BOX_MIN(3, x, 32) = (box.pmin.x);
        MAKE_BOX_MIN(3, y, 32) = (box.pmin.y);
        MAKE_BOX_MIN(3, z, 32) = (box.pmin.z);

        MAKE_BOX_MAX(3, x, 32) = (box.pmax.x);
        MAKE_BOX_MAX(3, y, 32) = (box.pmax.y);
        MAKE_BOX_MAX(3, z, 32) = (box.pmax.z);
    }

    uint node_id_ptr = MAKE_PTR(abvh_cnt, node_type::box32);
    abvh_cnt += 2;

    *(BoxNodeF32*)(&nodes[0] + DECODE_NODE(node_id_ptr)) = box_f32;
    return node_id_ptr;
}

uint BVHEncoder::pack_leaf(cl_amd_user_bvh_node_ptr current_node, std::vector<ABVHNode>& nodes, uint& abvh_cnt)
{
    uint encoded_ptr = CL_AMD_INVALID_NODE_PTR;

    ABVHNode abvh_leaf;
    if (callbacks_->pfn_get_node_type(current_node) == CL_AMD_BVH_NODE_TYPE_INSTANCE)
    {
        encoded_ptr = MAKE_PTR(abvh_cnt, node_type::object);
      cl_amd_transform tfm = callbacks_->pfn_get_transform(current_node);// 
     // transform_quick_inverse(callbacks_->pfn_get_transform(current_node)); //applying inverse on the app side, currently this call producing wrong output
        abvh_leaf.object.tfm = tfm;
        abvh_leaf.object.shape_id = callbacks_->pfn_get_instance_id(current_node);
        abvh_leaf.object.bottom_pointer = callbacks_->pfn_get_blas_offset(current_node);
        abvh_leaf.object.prim_offset = callbacks_->pfn_get_blas_prim_offset(current_node);
        nodes[DECODE_NODE(encoded_ptr)] = abvh_leaf;
        abvh_cnt++;

        //top_level_map.push_back(std::pair(callbacks_->pfn_get_blas(current_node), &nodes[DECODE_NODE(encoded_ptr)]));
        top_level_nodes_count_++;
    }
    else
    {
      if (callbacks_->pfn_get_node_type(current_node) == CL_AMD_BVH_NODE_TYPE_TRINAGLE) {
        int triangles_count = callbacks_->pfn_get_num_triangles(current_node);
        for (int i = 0; i < triangles_count; ++i) {
          abvh_leaf.tri = convert_to_gpu_triangle(callbacks_->pfn_get_triangle(current_node, i));
          //abvh_leaf.tri.triangles_count = abvh_leaf.tri.id;  // i == 0 ? triangles_count : 0;
          abvh_leaf.tri.id = 4;
          encoded_ptr = MAKE_PTR(abvh_cnt, triangle0);
          nodes[DECODE_NODE(encoded_ptr)] = abvh_leaf;
          abvh_cnt++;
        }
      }
        else
        {
            CustomNode custom_node;
            size_t data_size = callbacks_->pfn_get_user_data_size(current_node);
            char* data = (char*)callbacks_->pfn_get_user_data(current_node);
            memcpy(custom_node.data, data, data_size > MAX_DATA_SIZE ? MAX_DATA_SIZE : data_size);
            custom_node.size = data_size;
            encoded_ptr = MAKE_PTR(abvh_cnt, custom);
            *(CustomNode*)(&nodes[0] + DECODE_NODE(encoded_ptr)) = custom_node;
            abvh_cnt += 2;
        }
    }

    return encoded_ptr;
}

uint BVHEncoder::flatten_nodes(
    cl_amd_user_bvh_node_ptr current_node,
    std::vector<ABVHNode>& nodes,
    uint depth,
    uint& abvh_cnt,
    cl_amd_aabb current_aabb)
{
    #if 1
    uint encoded_ptr = CL_AMD_INVALID_NODE_PTR;
    if (callbacks_->pfn_get_node_type(current_node) != CL_AMD_BVH_NODE_TYPE_AABB)
    {
        return pack_leaf(current_node, nodes, abvh_cnt);
    }

    cl_amd_user_bvh_node_ptr children[4];
    cl_amd_user_bvh_node_ptr child0 = callbacks_->pfn_get_child(current_node, 0);
    cl_amd_user_bvh_node_ptr child1 = callbacks_->pfn_get_child(current_node, 1);

    bool valid_nodes[4] = { false, false, false, false };
    cl_amd_aabb bounds[4], childs_aabb[4];

    if (callbacks_->pfn_get_node_type(child0) != CL_AMD_BVH_NODE_TYPE_AABB)
    {
        children[0] = child0;
        childs_aabb[0] = callbacks_->pfn_get_child_aabb(current_node, 0);
        valid_nodes[0] = true;

        bounds[0].pmax = current_aabb.pmax;
        bounds[0].pmin = current_aabb.pmin;
    }
    else
    {
        children[0] = callbacks_->pfn_get_child(child0, 0);
        children[1] = callbacks_->pfn_get_child(child0, 1);
        childs_aabb[0] = callbacks_->pfn_get_child_aabb(child0, 0);
        childs_aabb[1] = callbacks_->pfn_get_child_aabb(child0, 1);
        valid_nodes[0] = true;
        valid_nodes[1] = true;

        bounds[0] = callbacks_->pfn_get_child_aabb(child0, 0);
        bounds[1] = callbacks_->pfn_get_child_aabb(child0, 1);
    }

    if (callbacks_->pfn_get_node_type(child1) != CL_AMD_BVH_NODE_TYPE_AABB)
    {
        children[2] = child1;
        valid_nodes[2] = true;
        childs_aabb[2] = callbacks_->pfn_get_child_aabb(current_node, 1);

        bounds[2].pmax = current_aabb.pmax;
        bounds[2].pmin = current_aabb.pmin;
    }
    else
    {
        children[2] = callbacks_->pfn_get_child(child1, 0);
        children[3] = callbacks_->pfn_get_child(child1, 1);
        valid_nodes[2] = true;
        valid_nodes[3] = true;
        childs_aabb[2] = callbacks_->pfn_get_child_aabb(child1, 0);
        childs_aabb[3] = callbacks_->pfn_get_child_aabb(child1, 1);

        bounds[2] = callbacks_->pfn_get_child_aabb(child1, 0);
        bounds[3] = callbacks_->pfn_get_child_aabb(child1, 1);
    }

    bool fp16_box = true;

    if (depth < FULL_PRECISION_DEPTH)
    {
        fp16_box = false;
    }

    bool is_root = depth == 0;
    /*if (fp16_box && is_root == false)
    {
        encoded_ptr = pack_box_node_f16(&nodes[0], valid_nodes, bounds, abvh_cnt);
    }
    else*/
    {
        encoded_ptr = pack_box_node_f32(&nodes[0], valid_nodes, bounds, is_root, abvh_cnt);
    }

    uint ptr = flatten_nodes(children[0], nodes, depth + 1, abvh_cnt, childs_aabb[0]);
    nodes[DECODE_NODE(encoded_ptr)].box.child0 = valid_nodes[0] ? ptr : CL_AMD_INVALID_NODE_PTR;

    if (valid_nodes[1])
    {
        ptr = flatten_nodes(children[1], nodes, depth + 1, abvh_cnt, childs_aabb[1]);

        nodes[DECODE_NODE(encoded_ptr)].box.child1 = ptr;
    }
    else
    {
        nodes[DECODE_NODE(encoded_ptr)].box.child1 = CL_AMD_INVALID_NODE_PTR;
    }

    ptr = flatten_nodes(children[2], nodes, depth + 1, abvh_cnt, childs_aabb[2]);
    nodes[DECODE_NODE(encoded_ptr)].box.child2 = valid_nodes[2] ? ptr : CL_AMD_INVALID_NODE_PTR;

    if (valid_nodes[3])
    {
        ptr = flatten_nodes(children[3], nodes, depth + 1, abvh_cnt, childs_aabb[3]);
        nodes[DECODE_NODE(encoded_ptr)].box.child3 = ptr;
    }
    else
    {
        nodes[DECODE_NODE(encoded_ptr)].box.child3 = CL_AMD_INVALID_NODE_PTR;
    }

    return encoded_ptr;
    #else

struct BVHStack {
    cl_amd_user_bvh_node_ptr node;
    uint child_number;
    uint parent_ptr;

    BVHStack(cl_amd_user_bvh_node_ptr in_node, uint in_child_number, uint in_parent_ptr)
        : node(in_node), child_number(in_child_number), parent_ptr(in_parent_ptr)
    {
    }
  };

#endif
}


cl_int BVHEncoder::copy_to_device(size_t size)
{    
    cl_int error = CL_SUCCESS;
    if (out_buffer_)
    {
        clEnqueueWriteBuffer(command_queue_, out_buffer_, CL_TRUE, 0, size, (void*)&out_buffer_cpu_[0],
            num_events_in_wait_list_, event_wait_list_, event_);
    }
    else
    {
        memcpy((void*)out_array_cpu_, (void*)&out_buffer_cpu_[0], size);
    }
    return error;
}

uint BVHEncoder::encode_bvh(
    cl_amd_user_bvh_node_ptr current_root,
    std::vector<ABVHNode>& nodes,
    uint& abvh_cnt)
{
    size_t num_nodes = subtree_size(BVH_STAT_NODE_COUNT, current_root, callbacks_);
    size_t num_leaf_nodes = subtree_size(BVH_STAT_LEAF_COUNT, current_root, callbacks_);
    assert(num_leaf_nodes <= num_nodes);
    size_t num_inner_nodes = num_nodes - num_leaf_nodes;

    int tentative_size = (num_leaf_nodes + 100) * 4;
    nodes.resize(tentative_size);

    if (callbacks_->pfn_get_node_type(current_root) != CL_AMD_BVH_NODE_TYPE_AABB)
    {
        ABVHNode current;
        nodes.push_back(current);
        return CL_SUCCESS;
    }

    uint depth = 0;

    cl_amd_aabb aabb_child0, aabb_child1, aabb_root;
    aabb_child0 = callbacks_->pfn_get_child_aabb(current_root, 0);
    aabb_child1 = callbacks_->pfn_get_child_aabb(current_root, 1);
    aabb_root = unite_aabb(aabb_child0, aabb_child1);

    return flatten_nodes(current_root, nodes, depth, abvh_cnt, aabb_root);

}

cl_int BVHEncoder::encode()
{
    out_buffer_cpu_.clear();
    std::vector<ABVHNode> abvh_nodes;
    uint current_size = 0, previous_size = 0, total_size = 0;

    // encode top level (if provided)
    encode_bvh(root_, abvh_nodes, current_size);

    total_size += current_size;
    previous_size = total_size;

    size_t amd_node_size = total_size * sizeof(ABVHNode);
    amd_node_size = align_up(amd_node_size, PADDING_SIZE);
    out_buffer_cpu_.resize(amd_node_size);
    memcpy(&out_buffer_cpu_[0], &abvh_nodes[0], amd_node_size);

    if (is_top_level) {
      BoxNodeF32 *root = (BoxNodeF32 *)(&out_buffer_cpu_[0]);
      root->padding0 = amd_node_size;  // root_offset
    }



    for (auto bvh_root : top_level_map)
    {
        std::vector<ABVHNode> abvh_nodes_bottom_level;
        current_size = 0;

        encode_bvh(bvh_root.first, abvh_nodes_bottom_level, current_size);
        total_size += current_size;
        amd_node_size = current_size * sizeof(ABVHNode);
        out_buffer_cpu_.resize(out_buffer_cpu_.size() + amd_node_size);
        memcpy(&out_buffer_cpu_[previous_size * sizeof(ABVHNode)], &abvh_nodes_bottom_level[0],
            amd_node_size);

        bvh_root.second->object.bottom_pointer = MAKE_PTR(previous_size, box32);
        previous_size = total_size;
    }

    //return copy_to_device(total_size * sizeof(ABVHNode));
    return 0;
}