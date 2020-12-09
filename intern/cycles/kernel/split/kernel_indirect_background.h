/*
 * Copyright 2011-2017 Blender Foundation
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

CCL_NAMESPACE_BEGIN

ccl_device void kernel_indirect_background(KernelGlobals *kg
#ifdef __KERNEL_OPENCL__
                                           ,
                                           ccl_constant KernelData *data,
                                           ccl_global void *split_data_buffer,
                                           ccl_global char *ray_state,
                                           KERNEL_BUFFER_PARAMS,
                                           ccl_global int *queue_index,
                                           ccl_global char *use_queues_flag,
                                           ccl_global unsigned int *work_pools,
                                           ccl_global float *buffer
#endif
)
{
  int thread_index = ccl_global_id(1) * ccl_global_size(0) + ccl_global_id(0);
  int ray_index;

  if (kernel_data.integrator.ao_bounces != INT_MAX) {
    ray_index = get_ray_index(kg,
                              thread_index,
                              QUEUE_ACTIVE_AND_REGENERATED_RAYS,
                              kernel_split_state_buffer(queue_data, int),
                              kernel_split_params.queue_size,
                              0);

    if (ray_index != QUEUE_EMPTY_SLOT) {
      if (IS_STATE(ray_state_buffer, ray_index, RAY_ACTIVE)) {
        ccl_global PathState *state = &kernel_split_state_buffer(path_state, PathState)[ray_index];
        if (path_state_ao_bounce(kg, state)) {
          kernel_split_path_end(kg, ray_state_buffer, ray_index);
        }
      }
    }
  }

  ray_index = get_ray_index(kg,
                            thread_index,
                            QUEUE_HITBG_BUFF_UPDATE_TOREGEN_RAYS,
                            kernel_split_state_buffer(queue_data, int),
                            kernel_split_params.queue_size,
                            0);

  if (ray_index == QUEUE_EMPTY_SLOT) {
    return;
  }

  if (IS_STATE(ray_state_buffer, ray_index, RAY_HIT_BACKGROUND)) {
    ccl_global PathState *state = &kernel_split_state_buffer(path_state, PathState)[ray_index];
    PathRadiance *L = &kernel_split_state_buffer_addr_space(path_radiance, PathRadiance)[ray_index];
    ccl_global Ray *ray = &kernel_split_state_buffer(ray, Ray)[ray_index];
    float3 throughput = kernel_split_state_buffer(throughput, float3)[ray_index];
    ShaderData *sd = kernel_split_sd(sd, ray_index);
    uint buffer_offset = kernel_split_state_buffer(buffer_offset, uint)[ray_index];
    ccl_global float *buffer = kernel_split_params.tile.buffer + buffer_offset;

    kernel_path_background(kg, state, ray, throughput, sd, buffer, L);
    kernel_split_path_end(kg, ray_state_buffer, ray_index);
  }
}

CCL_NAMESPACE_END
