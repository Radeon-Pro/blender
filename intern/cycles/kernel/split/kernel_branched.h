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

#ifdef __BRANCHED_PATH__

/* sets up the various state needed to do an indirect loop */
ccl_device_inline void kernel_split_branched_path_indirect_loop_init(KernelGlobals *kg,
#  ifdef __KERNEL_OPENCL__
                                                                     SPLIT_DATA_BUFFER_PARAMS,
                                                                     ccl_global char *ray_state,
#  endif
                                                                     int ray_index)
{
  SplitBranchedState *branched_state = &kernel_split_state_buffer_addr_space(
      branched_state, SplitBranchedState)[ray_index];

  /* save a copy of the state to restore later */
#  define BRANCHED_STORE(name, type) \
    branched_state->name = kernel_split_state_buffer(name, type)[ray_index]

  BRANCHED_STORE(path_state, PathState);
  BRANCHED_STORE(throughput, float3);
  BRANCHED_STORE(ray, Ray);
  BRANCHED_STORE(isect, Intersection);
  branched_state->ray_state = ray_state_buffer[ray_index];

  *kernel_split_sd(branched_state_sd, ray_index) = *kernel_split_sd(sd, ray_index);
  for (int i = 0; i < kernel_split_sd(branched_state_sd, ray_index)->num_closure; i++) {
    kernel_split_sd(branched_state_sd, ray_index)->closure[i] =
        kernel_split_sd(sd, ray_index)->closure[i];
  }

#  undef BRANCHED_STORE

  /* set loop counters to intial position */
  branched_state->next_closure = 0;
  branched_state->next_sample = 0;
}

/* ends an indirect loop and restores the previous state */
ccl_device_inline void kernel_split_branched_path_indirect_loop_end(KernelGlobals *kg,
#  ifdef __KERNEL_OPENCL__
                                                                    SPLIT_DATA_BUFFER_PARAMS,
                                                                    ccl_global char *ray_state,
#  endif
                                                                    int ray_index)
{
  SplitBranchedState *branched_state = &kernel_split_state_buffer_addr_space(
      branched_state, SplitBranchedState)[ray_index];

  /* restore state */
#  define BRANCHED_RESTORE(name, type) \
    kernel_split_state_buffer(name, type)[ray_index] = branched_state->name

  BRANCHED_RESTORE(path_state, PathState);
  BRANCHED_RESTORE(throughput, float3);
  BRANCHED_RESTORE(ray, Ray);
  BRANCHED_RESTORE(isect, Intersection);
  ray_state_buffer[ray_index] = branched_state->ray_state;

  *kernel_split_sd(sd, ray_index) = *kernel_split_sd(branched_state_sd, ray_index);
  for (int i = 0; i < kernel_split_sd(branched_state_sd, ray_index)->num_closure; i++) {
    kernel_split_sd(sd, ray_index)->closure[i] =
        kernel_split_sd(branched_state_sd, ray_index)->closure[i];
  }

#  undef BRANCHED_RESTORE

  /* leave indirect loop */
  REMOVE_RAY_FLAG(ray_state_buffer, ray_index, RAY_BRANCHED_INDIRECT);
}

ccl_device_inline bool kernel_split_branched_indirect_start_shared(KernelGlobals *kg,
#  ifdef __KERNEL_OPENCL__
                                                                   SPLIT_DATA_BUFFER_PARAMS,
                                                                   ccl_global char *ray_state,
#  endif
                                                                   int ray_index)
{
  int inactive_ray = dequeue_ray_index(QUEUE_INACTIVE_RAYS,
                                       kernel_split_state_buffer(queue_data, int),
                                       kernel_split_params.queue_size,
                                       kernel_split_params.queue_index);

  if (!IS_STATE(ray_state_buffer, inactive_ray, RAY_INACTIVE)) {
    return false;
  }

#  define SPLIT_DATA_ENTRY(type, name, num) \
    if (num) { \
      kernel_split_state_buffer(name, type)[inactive_ray] = kernel_split_state_buffer( \
          name, type)[ray_index]; \
    }
#  define SPLIT_DATA_ENTRY_ADDR_SPACE(type, name, num) \
    if (num) { \
      kernel_split_state_buffer_addr_space(name, type)[inactive_ray] = kernel_split_state_buffer_addr_space( \
          name, type)[ray_index]; \
    }

  SPLIT_DATA_ENTRY(float3, throughput, 1)
  SPLIT_DATA_ENTRY_ADDR_SPACE(PathRadiance, path_radiance, 1)
  SPLIT_DATA_ENTRY(Ray, ray, 1)
  SPLIT_DATA_ENTRY(PathState, path_state, 1)
  SPLIT_DATA_ENTRY(Intersection, isect, 1)
  SPLIT_DATA_ENTRY(BsdfEval, bsdf_eval, 1)
  SPLIT_DATA_ENTRY(int, is_lamp, 1)
  SPLIT_DATA_ENTRY(Ray, light_ray, 1)
  SPLIT_DATA_ENTRY(int, queue_data, (NUM_QUEUES * 2))
  SPLIT_DATA_ENTRY(uint, buffer_offset, 1)
  SPLIT_DATA_ENTRY_ADDR_SPACE(ShaderDataTinyStorage, sd_DL_shadow, 1)

#  ifdef __SUBSURFACE__
  SPLIT_DATA_ENTRY(SubsurfaceIndirectRays, ss_rays, 1)
#  endif /* __SUBSURFACE__ */

#  ifdef __VOLUME__
  SPLIT_DATA_ENTRY(PathState, state_shadow, 1)
#  endif /* __VOLUME__ */

  SPLIT_DATA_ENTRY_ADDR_SPACE(SplitBranchedState, branched_state, 1)
  SPLIT_DATA_ENTRY_ADDR_SPACE(ShaderData, _branched_state_sd, 0)

  SPLIT_DATA_ENTRY_ADDR_SPACE(ShaderData, _sd, 0)

#  undef SPLIT_DATA_ENTRY
#  undef SPLIT_DATA_ENTRY_ADDR_SPACE

  *kernel_split_sd(sd, inactive_ray) = *kernel_split_sd(sd, ray_index);
  for (int i = 0; i < kernel_split_sd(sd, ray_index)->num_closure; i++) {
    kernel_split_sd(sd, inactive_ray)->closure[i] = kernel_split_sd(sd, ray_index)->closure[i];
  }

  kernel_split_state_buffer_addr_space(branched_state, SplitBranchedState)[inactive_ray].shared_sample_count = 0;
  kernel_split_state_buffer_addr_space(branched_state, SplitBranchedState)[inactive_ray].original_ray = ray_index;
  kernel_split_state_buffer_addr_space(branched_state, SplitBranchedState)[inactive_ray].waiting_on_shared_samples = false;

  PathRadiance *L = &kernel_split_state_buffer_addr_space(path_radiance, PathRadiance)[ray_index];
  PathRadiance *inactive_L = &kernel_split_state_buffer_addr_space(path_radiance,
                                                                   PathRadiance)[inactive_ray];

  path_radiance_init(kg, inactive_L);
  path_radiance_copy_indirect(inactive_L, L);

  ray_state_buffer[inactive_ray] = RAY_REGENERATED;
  ADD_RAY_FLAG(ray_state_buffer, inactive_ray, RAY_BRANCHED_INDIRECT_SHARED);
  ADD_RAY_FLAG(ray_state_buffer, inactive_ray, IS_FLAG(ray_state_buffer, ray_index, RAY_BRANCHED_INDIRECT));

  atomic_fetch_and_inc_uint32(
      (ccl_global uint *)&kernel_split_state_buffer_addr_space(branched_state, SplitBranchedState)[ray_index].shared_sample_count);

  return true;
}

/* bounce off surface and integrate indirect light */
ccl_device_noinline bool kernel_split_branched_path_surface_indirect_light_iter(
    KernelGlobals *kg,
#  ifdef __KERNEL_OPENCL__
    SPLIT_DATA_BUFFER_PARAMS,
    ccl_global char *ray_state,
#  endif
    int ray_index,
    float num_samples_adjust,
    ShaderData *saved_sd,
    bool reset_path_state,
    bool wait_for_shared)
{
  SplitBranchedState *branched_state = &kernel_split_state_buffer_addr_space(
      branched_state, SplitBranchedState)[ray_index];

  ShaderData *sd = saved_sd;
  PathRadiance *L = &kernel_split_state_buffer_addr_space(path_radiance, PathRadiance)[ray_index];
  float3 throughput = branched_state->throughput;
  ccl_global PathState *ps = &kernel_split_state_buffer(path_state, PathState)[ray_index];

  float sum_sample_weight = 0.0f;
#  ifdef __DENOISING_FEATURES__
  if (ps->denoising_feature_weight > 0.0f) {
    for (int i = 0; i < sd->num_closure; i++) {
      const ShaderClosure *sc = &sd->closure[i];

      /* transparency is not handled here, but in outer loop */
      if (!CLOSURE_IS_BSDF(sc->type) || CLOSURE_IS_BSDF_TRANSPARENT(sc->type)) {
        continue;
      }

      sum_sample_weight += sc->sample_weight;
    }
  }
  else {
    sum_sample_weight = 1.0f;
  }
#  endif /* __DENOISING_FEATURES__ */

  for (int i = branched_state->next_closure; i < sd->num_closure; i++) {
    const ShaderClosure *sc = &sd->closure[i];

    if (!CLOSURE_IS_BSDF(sc->type))
      continue;
    /* transparency is not handled here, but in outer loop */
    if (sc->type == CLOSURE_BSDF_TRANSPARENT_ID)
      continue;

    int num_samples;

    if (CLOSURE_IS_BSDF_DIFFUSE(sc->type))
      num_samples = kernel_data.integrator.diffuse_samples;
    else if (CLOSURE_IS_BSDF_BSSRDF(sc->type))
      num_samples = 1;
    else if (CLOSURE_IS_BSDF_GLOSSY(sc->type))
      num_samples = kernel_data.integrator.glossy_samples;
    else
      num_samples = kernel_data.integrator.transmission_samples;

    num_samples = ceil_to_int(num_samples_adjust * num_samples);

    float num_samples_inv = num_samples_adjust / num_samples;

    for (int j = branched_state->next_sample; j < num_samples; j++) {
      if (reset_path_state) {
        *ps = branched_state->path_state;
      }

      ps->rng_hash = cmj_hash(branched_state->path_state.rng_hash, i);

      ccl_global float3 *tp = &kernel_split_state_buffer(throughput, float3)[ray_index];
      *tp = throughput;

      ccl_global Ray *bsdf_ray = &kernel_split_state_buffer(ray, Ray)[ray_index];

      if (!kernel_branched_path_surface_bounce(
              kg, sd, sc, j, num_samples, tp, ps, &L->state, bsdf_ray, sum_sample_weight)) {
        continue;
      }

      ps->rng_hash = branched_state->path_state.rng_hash;

      /* update state for next iteration */
      branched_state->next_closure = i;
      branched_state->next_sample = j + 1;

      /* start the indirect path */
      *tp *= num_samples_inv;

      if (kernel_split_branched_indirect_start_shared(kg,
#  ifdef __KERNEL_OPENCL__
                                                      SPLIT_DATA_BUFFER_ARGS,
                                                      ray_state,
#  endif
                                                      ray_index)) {
        continue;
      }

      return true;
    }

    branched_state->next_sample = 0;
  }

  branched_state->next_closure = sd->num_closure;

  if (wait_for_shared) {
    branched_state->waiting_on_shared_samples = (branched_state->shared_sample_count > 0);
    if (branched_state->waiting_on_shared_samples) {
      return true;
    }
  }

  return false;
}

#endif /* __BRANCHED_PATH__ */

CCL_NAMESPACE_END
