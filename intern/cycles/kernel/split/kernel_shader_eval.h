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

/* This kernel evaluates ShaderData structure from the values computed
 * by the previous kernels.
 */
ccl_device void kernel_shader_eval(KernelGlobals *kg
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

  int ray_index = ccl_global_id(1) * ccl_global_size(0) + ccl_global_id(0);
  /* Sorting on cuda split is not implemented */
#ifdef __KERNEL_CUDA__
  int queue_id = kernel_split_params.queue_index[QUEUE_ACTIVE_AND_REGENERATED_RAYS];
#else
  int queue_id = kernel_split_params.queue_index[QUEUE_SHADER_SORTED_RAYS];
#endif
  if (ray_index >= queue_id) {
    return;
  }
  ray_index = get_ray_index(kg,
                            ray_index,
#ifdef __KERNEL_CUDA__
                            QUEUE_ACTIVE_AND_REGENERATED_RAYS,
#else
                            QUEUE_SHADER_SORTED_RAYS,
#endif
                            kernel_split_state_buffer(queue_data, int),
                            kernel_split_params.queue_size,
                            0);

  if (ray_index == QUEUE_EMPTY_SLOT) {
    return;
  }

  if (IS_STATE(ray_state_buffer, ray_index, RAY_ACTIVE)) {
    ccl_global PathState *state = kernel_split_state_buffer(path_state, PathState) + ray_index;
    uint buffer_offset = kernel_split_state_buffer(buffer_offset, uint)[ray_index];
    ccl_global float *buffer = kernel_split_params.tile.buffer + buffer_offset;

    shader_eval_surface(kg, kernel_split_sd(sd, ray_index), state, buffer, state->flag);
#ifdef __BRANCHED_PATH__
    if (kernel_data.integrator.branched) {
      shader_merge_closures(kernel_split_sd(sd, ray_index));
    }
    else
#endif
    {
      shader_prepare_closures(kernel_split_sd(sd, ray_index), state);
    }
  }
}

CCL_NAMESPACE_END
