#pragma once

#include <mutex>

#include "BLI_string_multi_map.h"

#include "particles_state.hpp"

namespace BParticles {

using BLI::StringMultiMap;
using FN::AttributesRefGroup;

class ParticleAllocator : BLI::NonCopyable, BLI::NonMovable {
 private:
  ParticlesState &m_state;
  StringMultiMap<ParticleSet *> m_allocated_particles;
  std::mutex m_request_mutex;

 public:
  ParticleAllocator(ParticlesState &state);

  /**
   * Access all particles that have been allocated by this allocator.
   */
  StringMultiMap<ParticleSet *> allocated_particles();

  /**
   * Get memory buffers for new particles.
   */
  AttributesRefGroup request(StringRef particle_system_name, uint size);

 private:
  void initialize_new_particles(AttributesRefGroup &attributes_group);
};

/* ParticleAllocator inline functions
 ********************************************/

inline StringMultiMap<ParticleSet *> ParticleAllocator::allocated_particles()
{
  return m_allocated_particles;
}

}  // namespace BParticles
