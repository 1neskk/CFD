#include <device_launch_parameters.h>

#include "random.cuh"

namespace random_ns {
__global__ void init_random_states(curandState* state,
                                   unsigned long long seed) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    generator::init(state, seed, id);
}
}  // namespace random_ns