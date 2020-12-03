/*
 * Copyright 2011-2015 Blender Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define REMOVE_KERNEL_GLOBALS_ARGUMENT 0

#if REMOVE_KERNEL_GLOBALS_ARGUMENT

#  include "kernel/kernel_compat_opencl.h"
#  include "kernel/split/kernel_split_common.h"

#  define kernel_split_state (kg->split_data)
#  define kernel_split_params (kg->split_param_data)

#  if !defined(queue_data_OFFSET)
#    define queue_data_OFFSET 0
#  endif
#  if !defined(path_state_OFFSET)
#    define path_state_OFFSET 0
#  endif
#  if !defined(branched_state_OFFSET)
#    define branched_state_OFFSET 0
#  endif
#  if !defined(ray_OFFSET)
#    define ray_OFFSET 0
#  endif
#  if !defined(path_radiance_OFFSET)
#    define path_radiance_OFFSET 0
#  endif
#  if !defined(isect_OFFSET)
#    define isect_OFFSET 0
#  endif

#  define kernel_split_data_buffer(buf, type) \
    ((ccl_global type *)((ccl_global char *)split_data_buffer + buf##_OFFSET))

#  define kernel_split_data_buffer_addr_space(buf, type) \
    ((type *)((ccl_global char *)split_data_buffer + buf##_OFFSET))

ccl_device_inline void kernel_split_path_end_new(KernelGlobals *kg,
                                             ccl_global char *ray_state,
                                             int ray_index)
{
#  ifdef __BRANCHED_PATH__
#    ifdef __SUBSURFACE__
  ccl_addr_space SubsurfaceIndirectRays *ss_indirect = kernel_split_data_buffer(
                                                           ss_rays, SubsurfaceIndirectRays) +
                                                       ray_index;

  if (ss_indirect->num_rays) {
    ASSIGN_RAY_STATE(ray_state, ray_index, RAY_UPDATE_BUFFER);
  }
  else
#    endif /* __SUBSURFACE__ */
      if (IS_FLAG(ray_state, ray_index, RAY_BRANCHED_INDIRECT_SHARED)) {
    // int orig_ray = kernel_split_state.branched_state[ray_index].original_ray;
    int orig_ray =
        kernel_split_data_buffer(branched_state, SplitBranchedState)[ray_index].original_ray;

    // PathRadiance *L = &kernel_split_state.path_radiance[ray_index];
    PathRadiance *L = kernel_split_data_buffer_addr_space(path_radiance, PathRadiance) + ray_index;
    // PathRadiance *orig_ray_L = &kernel_split_state.path_radiance[orig_ray];
    PathRadiance *orig_ray_L = kernel_split_data_buffer_addr_space(path_radiance, PathRadiance) +
                               orig_ray;

    path_radiance_sum_indirect(L);
    path_radiance_accum_sample(orig_ray_L, L);

    // atomic_fetch_and_dec_uint32(
    //    (ccl_global uint *)&kernel_split_state.branched_state[orig_ray].shared_sample_count);
    atomic_fetch_and_dec_uint32(
        (ccl_global uint *)&kernel_split_data_buffer(branched_state, SplitBranchedState)[orig_ray]
            .shared_sample_count);

    ASSIGN_RAY_STATE(ray_state, ray_index, RAY_INACTIVE);
  }
  else if (IS_FLAG(ray_state, ray_index, RAY_BRANCHED_LIGHT_INDIRECT)) {
    ASSIGN_RAY_STATE(ray_state, ray_index, RAY_LIGHT_INDIRECT_NEXT_ITER);
  }
  else if (IS_FLAG(ray_state, ray_index, RAY_BRANCHED_VOLUME_INDIRECT)) {
    ASSIGN_RAY_STATE(ray_state, ray_index, RAY_VOLUME_INDIRECT_NEXT_ITER);
  }
  else if (IS_FLAG(ray_state, ray_index, RAY_BRANCHED_SUBSURFACE_INDIRECT)) {
    ASSIGN_RAY_STATE(ray_state, ray_index, RAY_SUBSURFACE_INDIRECT_NEXT_ITER);
  }
  else {
    ASSIGN_RAY_STATE(ray_state, ray_index, RAY_UPDATE_BUFFER);
  }
#  else
  ASSIGN_RAY_STATE(ray_state, ray_index, RAY_UPDATE_BUFFER);
#  endif
}

ccl_device void kernel_scene_intersect(KernelGlobals *kg,
                                       ccl_constant KernelData *data,
                                       ccl_global void *split_data_buffer,
                                       ccl_global char *ray_state,
                                       KERNEL_BUFFER_PARAMS,
                                       ccl_global int *queue_index,
                                       ccl_global char *use_queues_flag,
                                       ccl_global unsigned int *work_pools,
                                       ccl_global float *buffer)
{
  /* Fetch use_queues_flag */
  char local_use_queues_flag = *use_queues_flag;
  ccl_barrier(CCL_LOCAL_MEM_FENCE);

  int ray_index = ccl_global_id(1) * ccl_global_size(0) + ccl_global_id(0);
  if (local_use_queues_flag) {
    ray_index = get_ray_index(kg,
                              ray_index,
                              QUEUE_ACTIVE_AND_REGENERATED_RAYS,
                              kernel_split_data_buffer(queue_data, int),  // kernel_split_state.queue_data,
                              kernel_split_params.queue_size,
                              0);

    if (ray_index == QUEUE_EMPTY_SLOT) {
      return;
    }
  }

  /* All regenerated rays become active here */
  if (IS_STATE(/*kernel_split_state.*/ray_state, ray_index, RAY_REGENERATED)) {
#  ifdef __BRANCHED_PATH__
    if (kernel_split_data_buffer(branched_state, SplitBranchedState)[ray_index]
            .waiting_on_shared_samples
        /*kernel_split_state.branched_state[ray_index].waiting_on_shared_samples*/) {
      kernel_split_path_end_new(kg, ray_state, ray_index);
    }
    else
#  endif /* __BRANCHED_PATH__ */
    {
      ASSIGN_RAY_STATE(/*kernel_split_state.*/ray_state, ray_index, RAY_ACTIVE);
    }
  }

  if (!IS_STATE(/*kernel_split_state.*/ray_state, ray_index, RAY_ACTIVE)) {
    return;
  }

  //ccl_global PathState *state = &kernel_split_state.path_state[ray_index];
  ccl_global PathState *state = kernel_split_data_buffer(path_state, PathState) +
                                ray_index;
  //Ray ray = kernel_split_state.ray[ray_index];
  Ray ray = kernel_split_data_buffer(ray, Ray)[ray_index];
  //PathRadiance *L = &kernel_split_state.path_radiance[ray_index];
  PathRadiance *L = kernel_split_data_buffer_addr_space(path_radiance, PathRadiance) +
                    ray_index;

  Intersection isect;
  bool hit = kernel_path_scene_intersect(kg, state, &ray, &isect, L);
  //kernel_split_state.isect[ray_index] = isect;
  kernel_split_data_buffer(isect, Intersection)[ray_index] = isect;

  if (!hit) {
    /* Change the state of rays that hit the background;
     * These rays undergo special processing in the
     * background_bufferUpdate kernel.
     */
    ASSIGN_RAY_STATE(/*kernel_split_state.*/ray_state, ray_index, RAY_HIT_BACKGROUND);
  }
}

__kernel void kernel_ocl_path_trace_scene_intersect(ccl_global char *kg_global,
                                                    ccl_constant KernelData *data,
                                                    ccl_global void *split_data_buffer,
                                                    ccl_global char *ray_state,
                                                    KERNEL_BUFFER_PARAMS,
                                                    ccl_global int *queue_index,
                                                    ccl_global char *use_queues_flag,
                                                    ccl_global unsigned int *work_pools,
                                                    ccl_global float *buffer)
{
#  ifdef LOCALS_TYPE
  ccl_local LOCALS_TYPE locals;
#  endif

  KernelGlobals *kg = (KernelGlobals *)kg_global;
#  if 0
  if (ccl_local_id(0) + ccl_local_id(1) == 0) {
    kg->data = data;

    kernel_split_params.queue_index = queue_index;
    kernel_split_params.use_queues_flag = use_queues_flag;
    kernel_split_params.work_pools = work_pools;
    kernel_split_params.tile.buffer = buffer;

    split_data_init(kg,
                    &kernel_split_state,
                    ccl_global_size(0) * ccl_global_size(1),
                    split_data_buffer,
                    ray_state);
  }

  kernel_set_buffer_pointers(kg, KERNEL_BUFFER_ARGS);
#  endif

  kernel_scene_intersect(kg,
                         data,
                         split_data_buffer,
                         ray_state,
                         KERNEL_BUFFER_ARGS,
                         queue_index,
                         use_queues_flag,
                         work_pools,
                         buffer
#  ifdef LOCALS_TYPE
                         ,
                         &locals
#  endif
  );
}

#else

#include "kernel/kernel_compat_opencl.h"
#include "kernel/split/kernel_split_common.h"
#include "kernel/split/kernel_scene_intersect.h"

#define KERNEL_NAME scene_intersect
#include "kernel/kernels/opencl/kernel_split_function.h"
#undef KERNEL_NAME

#endif
