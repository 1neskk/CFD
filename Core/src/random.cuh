#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <glm/glm.hpp>
#include <random>

namespace random_ns {
class generator {
   public:
    __device__ static void init(curandState *state, unsigned long long seed,
                                int id) {
        curand_init(seed, id, 0, state);
    }

    __device__ static uint32_t uint(curandState *state, int id) {
        return curand(&state[id]);
    }

    __device__ static uint32_t uint(curandState *state, uint32_t min,
                                    uint32_t max, int id) {
        return min + (curand(&state[id]) % (max - min + 1));
    }

    __device__ static float get_float(curandState *state, int id) {
        return curand_uniform(&state[id]);
    }

    __device__ static float get_float(curandState *state, float min, float max,
                                      int id) {
        return min + (max - min) * curand_uniform(&state[id]);
    }

    __device__ static glm::vec3 vec3(curandState *state, int id) {
        return {get_float(state, id), get_float(state, id), get_float(state, id)};
    }

    __device__ static glm::vec3 vec3(curandState *state, float min, float max,
                                     int id) {
        return {get_float(state, min, max, id), get_float(state, min, max, id),
                get_float(state, min, max, id)};
    }

    __device__ static glm::vec3 in_unit_sphere(curandState *state, int id) {
        return glm::normalize(vec3(state, -1.0f, 1.0f, id));
    }

    __device__ static glm::vec3 pcg_in_unit_sphere(uint32_t &seed) {
        return glm::normalize(glm::vec3(pcg_float(seed) * 2.0f - 1.0f,
                                        pcg_float(seed) * 2.0f - 1.0f,
                                        pcg_float(seed) * 2.0f - 1.0f));
    }

    __device__ static uint32_t pcg_hash(uint32_t &seed) {
        uint32_t state = seed * 747796405u + 2891336453u;
        uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        return (word >> 22u) ^ word;
    }

    __device__ static float pcg_float(uint32_t &seed) {
        seed = pcg_hash(seed);
        return static_cast<float>(seed) / static_cast<float>(UINT32_MAX);
    }

    __device__ static glm::vec2 pcg_in_unit_disk(uint32_t &seed) {
        glm::vec2 p;
        do {
            p = 2.0f * glm::vec2(pcg_float(seed), pcg_float(seed)) -
                glm::vec2(1.0f, 1.0f);
        } while (glm::dot(p, p) >= 1.0f);
        return p;
    }
};
}  // namespace random_ns
