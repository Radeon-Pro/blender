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

#if defined(__BRANCHED_PATH__) && defined(__SUBSURFACE__)

ccl_device_inline void kernel_split_branched_path_subsurface_indirect_light_init(
    KernelGlobals *kg,
#  ifdef __KERNEL_OPENCL__
    SPLIT_DATA_BUFFER_PARAMS,
    ccl_global char *ray_state,
#  endif
    int ray_index)
{
  kernel_split_branched_path_indirect_loop_init(kg,
#  ifdef __KERNEL_OPENCL__
                                                SPLIT_DATA_BUFFER_ARGS,
                                                ray_state,
#  endif
                                                ray_index);

  SplitBranchedState *branched_state = &kernel_split_state_buffer_addr_space(
      branched_state, SplitBranchedState)[ray_index];

  branched_state->ss_next_closure = 0;
  branched_state->ss_next_sample = 0;

  branched_state->num_hits = 0;
  branched_state->next_hit = 0;

  ADD_RAY_FLAG(ray_state_buffer, ray_index, RAY_BRANCHED_SUBSURFACE_INDIRECT);
}

ccl_device_noinline bool kernel_split_branched_path_subsurface_indirect_light_iter(
    KernelGlobals *kg,
#  ifdef __KERNEL_OPENCL__
    SPLIT_DATA_BUFFER_PARAMS,
#  endif
    int ray_index)
{
  SplitBranchedState *branched_state = &kernel_split_state_buffer_addr_space(
      branched_state, SplitBranchedState)[ray_index];

  ShaderData *sd = kernel_split_sd(branched_state_sd, ray_index);
  PathRadiance *L = &kernel_split_state_buffer_addr_space(path_radiance, PathRadiance)[ray_index];
  ShaderData *emission_sd = AS_SHADER_DATA(
      &kernel_split_state_buffer_addr_space(sd_DL_shadow, ShaderDataTinyStorage)[ray_index]);

  for (int i = branched_state->ss_next_closure; i < sd->num_closure; i++) {
    ShaderClosure *sc = &sd->closure[i];

    if (!CLOSURE_IS_BSSRDF(sc->type))
      continue;

    /* Closure memory will be overwritten, so read required variables now. */
    Bssrdf *bssrdf = (Bssrdf *)sc;
    ClosureType bssrdf_type = sc->type;
    float bssrdf_roughness = bssrdf->roughness;

    /* set up random number generator */
    if (branched_state->ss_next_sample == 0 && branched_state->next_hit == 0 &&
        branched_state->next_closure == 0 && branched_state->next_sample == 0) {
      branched_state->lcg_state = lcg_state_init_addrspace(&branched_state->path_state,
                                                           0x68bc21eb);
    }
    int num_samples = kernel_data.integrator.subsurface_samples * 3;
    float num_samples_inv = 1.0f / num_samples;
    uint bssrdf_rng_hash = cmj_hash(branched_state->path_state.rng_hash, i);

    /* do subsurface scatter step with copy of shader data, this will
     * replace the BSSRDF with a diffuse BSDF closure */
    for (int j = branched_state->ss_next_sample; j < num_samples; j++) {
      ccl_global PathState *hit_state = &kernel_split_state_buffer(path_state,
                                                                   PathState)[ray_index];
      *hit_state = branched_state->path_state;
      hit_state->rng_hash = bssrdf_rng_hash;
      path_state_branch(hit_state, j, num_samples);

      ccl_global LocalIntersection *ss_isect = &branched_state->ss_isect;
      float bssrdf_u, bssrdf_v;
      path_branched_rng_2D(
          kg, bssrdf_rng_hash, hit_state, j, num_samples, PRNG_BSDF_U, &bssrdf_u, &bssrdf_v);

      /* intersection is expensive so avoid doing multiple times for the same input */
      if (branched_state->next_hit == 0 && branched_state->next_closure == 0 &&
          branched_state->next_sample == 0) {
        uint lcg_state = branched_state->lcg_state;
        LocalIntersection ss_isect_private;

        branched_state->num_hits = subsurface_scatter_multi_intersect(
            kg, &ss_isect_private, sd, hit_state, sc, &lcg_state, bssrdf_u, bssrdf_v, true);

        branched_state->lcg_state = lcg_state;
        *ss_isect = ss_isect_private;
      }

      hit_state->rng_offset += PRNG_BOUNCE_NUM;

#  ifdef __VOLUME__
      Ray volume_ray = branched_state->ray;
      bool need_update_volume_stack = kernel_data.integrator.use_volumes &&
                                      sd->object_flag & SD_OBJECT_INTERSECTS_VOLUME;
#  endif /* __VOLUME__ */

      /* compute lighting with the BSDF closure */
      for (int hit = branched_state->next_hit; hit < branched_state->num_hits; hit++) {
        ShaderData *bssrdf_sd = kernel_split_sd(sd, ray_index);
        *bssrdf_sd = *sd; /* note: copy happens each iteration of inner loop, this is
                           * important as the indirect path will write into bssrdf_sd */

        LocalIntersection ss_isect_private = *ss_isect;
        subsurface_scatter_multi_setup(
            kg, &ss_isect_private, hit, bssrdf_sd, hit_state, bssrdf_type, bssrdf_roughness);
        *ss_isect = ss_isect_private;

#  ifdef __VOLUME__
        if (need_update_volume_stack) {
          /* Setup ray from previous surface point to the new one. */
          float3 P = ray_offset(bssrdf_sd->P, -bssrdf_sd->Ng);
          volume_ray.D = normalize_len(P - volume_ray.P, &volume_ray.t);

          for (int k = 0; k < VOLUME_STACK_SIZE; k++) {
            hit_state->volume_stack[k] = branched_state->path_state.volume_stack[k];
          }

          kernel_volume_stack_update_for_subsurface(
              kg, emission_sd, &volume_ray, hit_state->volume_stack);
        }
#  endif /* __VOLUME__ */

#  ifdef __EMISSION__
        if (branched_state->next_closure == 0 && branched_state->next_sample == 0) {
          /* direct light */
          if (kernel_data.integrator.use_direct_light) {
            int all = (kernel_data.integrator.sample_all_lights_direct) ||
                      (hit_state->flag & PATH_RAY_SHADOW_CATCHER);
            kernel_branched_path_surface_connect_light(kg,
#    if defined(__SPLIT_KERNEL__) && defined(__VOLUME__)
                &kernel_split_state_buffer(
                    state_shadow,
                    PathState)[ccl_global_id(1) * ccl_global_size(0) + ccl_global_id(0)],
#    endif
                                                       bssrdf_sd,
                                                       emission_sd,
                                                       hit_state,
                                                       branched_state->throughput,
                                                       num_samples_inv,
                                                       L,
                                                       all);
          }
        }
#  endif /* __EMISSION__ */

        /* indirect light */
        if (kernel_split_branched_path_surface_indirect_light_iter(
                kg,
#  ifdef __KERNEL_OPENCL__
                                                                   SPLIT_DATA_BUFFER_ARGS,
                                                                   ray_state,
#  endif
                                                                   ray_index,
                                                                   num_samples_inv,
                                                                   bssrdf_sd,
                                                                   false,
                                                                   false)) {
          branched_state->ss_next_closure = i;
          branched_state->ss_next_sample = j;
          branched_state->next_hit = hit;

          return true;
        }

        branched_state->next_closure = 0;
      }

      branched_state->next_hit = 0;
    }

    branched_state->ss_next_sample = 0;
  }

  branched_state->ss_next_closure = sd->num_closure;

  branched_state->waiting_on_shared_samples = (branched_state->shared_sample_count > 0);
  if (branched_state->waiting_on_shared_samples) {
    return true;
  }

  kernel_split_branched_path_indirect_loop_end(kg,
#  ifdef __KERNEL_OPENCL__
                                               SPLIT_DATA_BUFFER_ARGS,
                                               ray_state_buffer,
#  endif
                                               ray_index);

  return false;
}

#endif /* __BRANCHED_PATH__ && __SUBSURFACE__ */

ccl_device void kernel_subsurface_scatter(KernelGlobals *kg
#ifdef __KERNEL_OPENCL__
                                          ,
                                          ccl_constant KernelData *data,
                                          SPLIT_DATA_BUFFER_PARAMS,
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
  if (thread_index == 0) {
    /* We will empty both queues in this kernel. */
    kernel_split_params.queue_index[QUEUE_ACTIVE_AND_REGENERATED_RAYS] = 0;
    kernel_split_params.queue_index[QUEUE_HITBG_BUFF_UPDATE_TOREGEN_RAYS] = 0;
  }

  int ray_index = ccl_global_id(1) * ccl_global_size(0) + ccl_global_id(0);
  ray_index = get_ray_index(kg,
                            ray_index,
                            QUEUE_ACTIVE_AND_REGENERATED_RAYS,
                            kernel_split_state_buffer(queue_data, int),
                            kernel_split_params.queue_size,
                            1);
  get_ray_index(kg,
                thread_index,
                QUEUE_HITBG_BUFF_UPDATE_TOREGEN_RAYS,
                kernel_split_state_buffer(queue_data, int),
                kernel_split_params.queue_size,
                1);

#ifdef __SUBSURFACE__
  if (IS_STATE(ray_state_buffer, ray_index, RAY_ACTIVE)) {
    ccl_global PathState *state = &kernel_split_state_buffer(path_state, PathState)[ray_index];
    PathRadiance *L = &kernel_split_state_buffer_addr_space(path_radiance, PathRadiance)[ray_index];
    ccl_global Ray *ray = &kernel_split_state_buffer(ray, Ray)[ray_index];
    ccl_global float3 *throughput = &kernel_split_state_buffer(throughput, float3)[ray_index];
    ccl_global SubsurfaceIndirectRays *ss_indirect = &kernel_split_state_buffer(
        ss_rays, SubsurfaceIndirectRays)[ray_index];
    ShaderData *sd = kernel_split_sd(sd, ray_index);
    ShaderData *emission_sd = AS_SHADER_DATA(&kernel_split_state_buffer_addr_space(sd_DL_shadow, ShaderDataTinyStorage)[ray_index]);

    if (sd->flag & SD_BSSRDF) {

#  ifdef __BRANCHED_PATH__
      if (!kernel_data.integrator.branched ||
          IS_FLAG(ray_state_buffer, ray_index, RAY_BRANCHED_INDIRECT)) {
#  endif
        if (kernel_path_subsurface_scatter(
                kg,
#  if defined(__SPLIT_KERNEL__) && defined(__VOLUME__)
                &kernel_split_state_buffer(
                    state_shadow,
                    PathState)[ccl_global_id(1) * ccl_global_size(0) + ccl_global_id(0)],
#  endif
                sd,
                emission_sd,
                L,
                state,
                ray,
                throughput,
                ss_indirect)) {
          kernel_split_path_end(kg,
#  ifdef __KERNEL_OPENCL__
                                SPLIT_DATA_BUFFER_ARGS,
#  endif
                                ray_state_buffer,
                                ray_index);
        }
#  ifdef __BRANCHED_PATH__
      }
      else {
        kernel_split_branched_path_subsurface_indirect_light_init(
            kg,
#    ifdef __KERNEL_OPENCL__
            SPLIT_DATA_BUFFER_ARGS,
            ray_state_buffer,
#    endif
            ray_index);

        if (kernel_split_branched_path_subsurface_indirect_light_iter(
                kg,
#ifdef __KERNEL_OPENCL__
                SPLIT_DATA_BUFFER_ARGS,
                ray_state_buffer,
#    endif
                ray_index)) {
          ASSIGN_RAY_STATE(ray_state_buffer, ray_index, RAY_REGENERATED);
        }
      }
#  endif
    }
  }

#  ifdef __BRANCHED_PATH__
  if (ccl_global_id(0) == 0 && ccl_global_id(1) == 0) {
    kernel_split_params.queue_index[QUEUE_SUBSURFACE_INDIRECT_ITER] = 0;
  }

  /* iter loop */
  ray_index = get_ray_index(kg,
                            ccl_global_id(1) * ccl_global_size(0) + ccl_global_id(0),
                            QUEUE_SUBSURFACE_INDIRECT_ITER,
                            kernel_split_state_buffer(queue_data, int),
                            kernel_split_params.queue_size,
                            1);

  if (IS_STATE(ray_state_buffer, ray_index, RAY_SUBSURFACE_INDIRECT_NEXT_ITER)) {
    /* for render passes, sum and reset indirect light pass variables
     * for the next samples */
    path_radiance_sum_indirect(&kernel_split_state_buffer_addr_space(path_radiance, PathRadiance)[ray_index]);
    path_radiance_reset_indirect(&kernel_split_state_buffer_addr_space(path_radiance, PathRadiance)[ray_index]);

    if (kernel_split_branched_path_subsurface_indirect_light_iter(
            kg,
#    ifdef __KERNEL_OPENCL__
            SPLIT_DATA_BUFFER_ARGS,
#    endif
            ray_index)) {
      ASSIGN_RAY_STATE(ray_state_buffer, ray_index, RAY_REGENERATED);
    }
  }
#  endif /* __BRANCHED_PATH__ */

#endif /* __SUBSURFACE__ */
}

CCL_NAMESPACE_END
