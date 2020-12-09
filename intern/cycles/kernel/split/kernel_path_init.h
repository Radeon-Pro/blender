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

/* This kernel initializes structures needed in path-iteration kernels.
 * This is the first kernel in ray-tracing logic.
 *
 * Ray state of rays outside the tile-boundary will be marked RAY_INACTIVE
 */
ccl_device void kernel_path_init(KernelGlobals *kg
#ifdef __KERNEL_OPENCL__
                                 ,
                                 ccl_constant KernelData *data,
                                 ccl_global void *split_data_buffer,
                                 ccl_global char *ray_state,
                                 KERNEL_BUFFER_PARAMS,
                                 ccl_global int *queue_index,
                                 ccl_global char *use_queues_flag,
                                 ccl_global unsigned int *work_pools_buffer,
                                 ccl_global float *buffer
#endif
)
{
  int ray_index = ccl_global_id(0) + ccl_global_id(1) * ccl_global_size(0);

  /* This is the first assignment to ray_state;
   * So we dont use ASSIGN_RAY_STATE macro.
   */
  ray_state_buffer[ray_index] = RAY_ACTIVE;

  /* Get work. */
  ccl_global uint *work_pools = kernel_split_params.work_pools;
  uint total_work_size = kernel_split_params.total_work_size;
  uint work_index;

  if (!get_next_work(kg, work_pools, total_work_size, ray_index, &work_index)) {
    /* No more work, mark ray as inactive */
    ray_state_buffer[ray_index] = RAY_INACTIVE;

    return;
  }

  ccl_global WorkTile *tile = &kernel_split_params.tile;
  uint x, y, sample;
  get_work_pixel(tile, work_index, &x, &y, &sample);

  /* Store buffer offset for writing to passes. */
  uint buffer_offset = (tile->offset + x + y * tile->stride) * kernel_data.film.pass_stride;
  kernel_split_state_buffer(buffer_offset, uint)[ray_index] = buffer_offset;

  /* Initialize random numbers and ray. */
  uint rng_hash;
  kernel_path_trace_setup(kg, sample, x, y, &rng_hash, &kernel_split_state_buffer(ray, Ray)[ray_index]);

  if (kernel_split_state_buffer(ray, Ray)[ray_index].t != 0.0f) {
    /* Initialize throughput, path radiance, Ray, PathState;
     * These rays proceed with path-iteration.
     */
    kernel_split_state_buffer(throughput, float3)[ray_index] = make_float3(1.0f, 1.0f, 1.0f);
    path_radiance_init(kg, &kernel_split_state_buffer_addr_space(path_radiance, PathRadiance)[ray_index]);
    path_state_init(kg,
                    AS_SHADER_DATA(&kernel_split_state_buffer_addr_space(sd_DL_shadow, ShaderDataTinyStorage)[ray_index]),
                    &kernel_split_state_buffer(path_state, PathState)[ray_index],
                    rng_hash,
                    sample,
                    &kernel_split_state_buffer(ray, Ray)[ray_index]);
#ifdef __SUBSURFACE__
    kernel_path_subsurface_init_indirect(&kernel_split_state_buffer(ss_rays, SubsurfaceIndirectRays)[ray_index]);
#endif
  }
  else {
    ASSIGN_RAY_STATE(ray_state_buffer, ray_index, RAY_TO_REGENERATE);
  }
}

CCL_NAMESPACE_END
