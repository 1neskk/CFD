#pragma once

#include <cuda_runtime.h>
#include "cuda_buffer.cuh"

class lbm_solver {
public:
    lbm_solver(int width, int height);
    ~lbm_solver();

    void init();
    void step();
    void reset();

    struct Rect {
        int x, y, w, h;
    };

    void add_solid(const Rect& rect);
    void register_external_density(int fd, size_t size);
    void register_external_velocity(int fd, size_t size);

    // Getters for visualization/interop
    cuda_buffer<float>& get_density() { return d_rho; }
    cuda_buffer<float2>& get_velocity() { return d_u; }
    cuda_buffer<unsigned char>& get_solid_mask() { return d_solid; }

    int get_width() const { return width; }
    int get_height() const { return height; }

private:
    int width;
    int height;
    size_t num_cells;

    // Simulation parameters
    float tau = 0.6f; // Lattice relaxation time
    float inlet_velocity = 0.1f;
    
    // Device pointers
    cuda_buffer<float> d_f;      // Distribution functions (current)
    cuda_buffer<float> d_f_new;  // Distribution functions (next)
    cuda_buffer<float> d_rho;    // Macroscopic density
    cuda_buffer<float2> d_u;     // Macroscopic velocity (x, y)
    // cuda_buffer<float> d_u_mag; // Velocity magnitude (optional, for vis) - Removed for now
    cuda_buffer<unsigned char> d_solid; // Solid mask (0: fluid, 1: solid)

    dim3 block_size;
    dim3 grid_size;

    cudaExternalMemory_t m_external_density = nullptr;
    cudaExternalMemory_t m_external_velocity = nullptr;
};
