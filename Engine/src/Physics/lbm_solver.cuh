#pragma once

#include <cuda_runtime.h>

#include "Utils/mesh_voxelizer.h"
#include "cuda_buffer.cuh"
#include "cuda_math.cuh"

struct settings {
    float tau = 0.55f;
    float inlet_velocity = 0.08f;
    float omega_shear = 1.98f;
    float omega_bulk = 1.90f;
    // float omega_high = 1.99f;
};

class lbm_solver {
   public:
    lbm_solver(int width, int height, int depth);
    ~lbm_solver();

    void init();
    void step();
    void reset();
    void compute_curl();

    struct Rect {
        int x, y, z, w, h, d;
    };

    __host__ __device__ void add_solid(const Rect& rect);
    void register_external_density(int fd, size_t size);
    void register_external_velocity(int fd, size_t size);
    void register_external_curl(int fd, size_t size);
    void register_external_solid(int fd, size_t size);

    void load_mesh(const std::string& path);

    // Getters for visualization/interop
    __host__ __device__ cuda_buffer<float>& get_density() { return d_rho; }
    __host__ __device__ cuda_buffer<cuda_math::vec4>& get_velocity() {
        return d_u;
    }
    __host__ __device__ cuda_buffer<unsigned char>& get_solid_mask() {
        return d_solid;
    }

    int get_width() const { return m_width; }
    int get_height() const { return m_height; }
    int get_depth() const { return m_depth; }
    settings& get_settings() { return m_settings; }

   private:
    int m_width;
    int m_height;
    int m_depth;
    size_t m_num_cells;

    settings m_settings;

    // Device pointers
    cuda_buffer<float> d_f;      // Distribution functions (current)
    cuda_buffer<float> d_f_new;  // Distribution functions (next)
    cuda_buffer<float> d_rho;    // Macroscopic density
    cuda_buffer<cuda_math::vec4>
        d_u;  // Macroscopic velocity (x, y, z, padding)
    cuda_buffer<unsigned char> d_solid;  // Solid mask (0: fluid, 1: solid)

    dim3 m_block_size;
    dim3 m_grid_size;

    cudaExternalMemory_t m_external_density = nullptr;
    cudaExternalMemory_t m_external_velocity = nullptr;
    cudaExternalMemory_t m_external_curl = nullptr;
    cudaExternalMemory_t m_external_solid = nullptr;

    cuda_buffer<float> d_curl;

    MeshVoxelizer* m_voxelizer;
};
