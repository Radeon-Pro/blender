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

CCL_NAMESPACE_BEGIN

/* Shadow ray cast for AO. */
ccl_device void kernel_shadow_blocked_ao(KernelGlobals *kg
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
  unsigned int ao_queue_length = kernel_split_params.queue_index[QUEUE_SHADOW_RAY_CAST_AO_RAYS];
  ccl_barrier(CCL_LOCAL_MEM_FENCE);

  int ray_index = QUEUE_EMPTY_SLOT;
  int thread_index = ccl_global_id(1) * ccl_global_size(0) + ccl_global_id(0);
  if (thread_index < ao_queue_length) {
    ray_index = get_ray_index(kg,
                              thread_index,
                              QUEUE_SHADOW_RAY_CAST_AO_RAYS,
                              kernel_split_state_buffer(queue_data, int),
                              kernel_split_params.queue_size,
                              1);
  }

  if (ray_index == QUEUE_EMPTY_SLOT) {
    return;
  }

  ShaderData *sd = kernel_split_sd(sd, ray_index);
  ShaderData *emission_sd = AS_SHADER_DATA(
      &kernel_split_state_buffer_addr_space(sd_DL_shadow, ShaderDataTinyStorage)[ray_index]);
  PathRadiance *L = &kernel_split_state_buffer_addr_space(path_radiance, PathRadiance)[ray_index];
  ccl_global PathState *state = &kernel_split_state_buffer(path_state, PathState)[ray_index];
  float3 throughput = kernel_split_state_buffer(throughput, float3)[ray_index];

#ifdef __BRANCHED_PATH__
  if (!kernel_data.integrator.branched ||
      IS_FLAG(ray_state_buffer, ray_index, RAY_BRANCHED_INDIRECT)) {
#endif
    kernel_path_ao(kg,
#ifdef __SPLIT_KERNEL__
                   &kernel_split_state_buffer(state_shadow, PathState)[thread_index],
#endif
                   sd,
                   emission_sd,
                   L,
                   state,
                   throughput,
                   shader_bsdf_alpha(kg, sd));
#ifdef __BRANCHED_PATH__
  }
  else {
    kernel_branched_path_ao(kg,
#  ifdef __SPLIT_KERNEL__
                            &kernel_split_state_buffer(state_shadow, PathState)[thread_index],
#  endif
                            sd,
                            emission_sd,
                            L,
                            state,
                            throughput);
  }
#endif
}

CCL_NAMESPACE_END
