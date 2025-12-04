#include "lbm_solver.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>

// D2Q9 Constants
__constant__ float w[9] = {
    4.0f / 9.0f,
    1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f,
    1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f
};

__constant__ int cx[9] = { 0, 1, 0, -1, 0, 1, -1, -1, 1 };
__constant__ int cy[9] = { 0, 0, 1, 0, -1, 1, 1, -1, -1 };

// Inverse directions for bounce-back
__constant__ int inv_dir[9] = { 0, 3, 4, 1, 2, 7, 8, 5, 6 };

__global__ void k_init(float* f, float* rho, float2* u, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x >= width || y >= height) return;

    // Initial conditions: rho = 1.0, u = 0
    float r = 1.0f;
    float u_x = 0.0f;
    float u_y = 0.0f;

    // Compute equilibrium
    float u_sq = u_x * u_x + u_y * u_y;
    for (int k = 0; k < 9; k++) {
        float cu = cx[k] * u_x + cy[k] * u_y;
        float f_eq = w[k] * r * (1.0f + 3.0f * cu + 4.5f * cu * cu - 1.5f * u_sq);
        f[k * width * height + idx] = f_eq;
    }

    rho[idx] = r;
    u[idx] = make_float2(u_x, u_y);
}

__global__ void k_stream_collide(
    const float* __restrict__ f_in, float* __restrict__ f_out, 
    float* __restrict__ rho_out, float2* __restrict__ u_out, 
    const unsigned char* __restrict__ solid,
    int width, int height, float tau, float inlet_velocity
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;
    int num_cells = width * height;

    if (x >= width || y >= height) return;

    // 1. Stream & Bounce-back
    float f_curr[9];
    bool is_solid = (solid[idx] > 0);

    if (is_solid) {
        rho_out[idx] = 0.0f;
        u_out[idx] = make_float2(0.0f, 0.0f);

        for (int k = 0; k < 9; k++) {
            f_out[k * num_cells + idx] = f_in[k * num_cells + idx];
        }
        return; 
    }

    // Fluid cell processing
    float rho = 0.0f;
    float ux = 0.0f;
    float uy = 0.0f;

    for (int k = 0; k < 9; k++) {
        int nx = x - cx[k];
        int ny = y - cy[k];

        // Periodic boundaries for Y only
        if (ny < 0) ny += height;
        if (ny >= height) ny -= height;

        // Inlet/Outlet for X
        if (nx < 0) {
            // Inlet: Equilibrium with inlet_velocity
            float r = 1.0f;
            float u_in_x = inlet_velocity;
            float u_in_y = 0.0f;
            float u_sq = u_in_x * u_in_x + u_in_y * u_in_y;
            float cu = cx[k] * u_in_x + cy[k] * u_in_y;
            f_curr[k] = w[k] * r * (1.0f + 3.0f * cu + 4.5f * cu * cu - 1.5f * u_sq);
        } else if (nx >= width) {
            // Outlet: Copy from right ghost cell
            int safe_nx = width - 1;
            int n_idx = ny * width + safe_nx;
            f_curr[k] = f_in[k * num_cells + n_idx];
        } else {
            int n_idx = ny * width + nx;

            // Check if neighbor is solid
            if (solid[n_idx] > 0) {
                int inv_k = inv_dir[k];
                f_curr[k] = f_in[inv_k * num_cells + idx];
            } else {
                f_curr[k] = f_in[k * num_cells + n_idx];
            }
        }

        rho += f_curr[k];
        ux += f_curr[k] * cx[k];
        uy += f_curr[k] * cy[k];
    }

    // 2. Macroscopic moments
    if (rho > 0.0f) {
        ux /= rho;
        uy /= rho;
    }

    rho_out[idx] = rho;
    u_out[idx] = make_float2(ux, uy);

    // 3. Collide (BGK)
    float u_sq = ux * ux + uy * uy;
    float omega = 1.0f / tau;

    for (int k = 0; k < 9; k++) {
        float cu = cx[k] * ux + cy[k] * uy;
        float f_eq = w[k] * rho * (1.0f + 3.0f * cu + 4.5f * cu * cu - 1.5f * u_sq);
        
        f_out[k * num_cells + idx] = f_curr[k] * (1.0f - omega) + f_eq * omega;
    }
}

lbm_solver::lbm_solver(int width, int height) : width(width), height(height) {
    num_cells = width * height;
    
    d_f = cuda_buffer<float>(9 * num_cells);
    d_f_new = cuda_buffer<float>(9 * num_cells);
    d_rho = cuda_buffer<float>(num_cells);
    d_u = cuda_buffer<float2>(num_cells);
    d_curl = cuda_buffer<float>(num_cells);
    d_solid = cuda_buffer<unsigned char>(num_cells);

    // Zero out solid mask initially
    d_solid.memset(0);

    block_size = dim3(16, 16);
    grid_size = dim3((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);
}

lbm_solver::~lbm_solver() {
    d_f.release();
    d_f_new.release();
    d_rho.release();
    d_u.release();
    d_curl.release();
    d_solid.release();
    
    if (m_external_density) {
        cudaDestroyExternalMemory(m_external_density);
    }
    if (m_external_velocity) {
        cudaDestroyExternalMemory(m_external_velocity);
    }
    if (m_external_curl) {
        cudaDestroyExternalMemory(m_external_curl);
    }
    if (m_external_solid) {
        cudaDestroyExternalMemory(m_external_solid);
    }
}

void lbm_solver::init() {
    k_init<<<grid_size, block_size>>>(d_f.get_data(), d_rho.get_data(), d_u.get_data(), width, height);
    cudaDeviceSynchronize();
}

void lbm_solver::step() {
    k_stream_collide<<<grid_size, block_size>>>(d_f.get_data(), d_f_new.get_data(), d_rho.get_data(), d_u.get_data(), d_solid.get_data(), width, height, tau, inlet_velocity);
    cudaDeviceSynchronize();

    compute_curl();

    std::swap(d_f, d_f_new);
}

void lbm_solver::reset() {
    init();
    d_solid.memset(0);
}

__global__ void k_add_solid_rect(unsigned char* solid, int width, int height, int rx, int ry, int rw, int rh) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x >= width || y >= height) return;

    if (x >= rx && x < rx + rw && y >= ry && y < ry + rh) {
        solid[idx] = 255;
    }
}

void lbm_solver::add_solid(const Rect& rect) {
    k_add_solid_rect<<<grid_size, block_size>>>(d_solid.get_data(), width, height, rect.x, rect.y, rect.w, rect.h);
    cudaDeviceSynchronize();
}

void lbm_solver::register_external_density(int fd, size_t size) {
    cudaExternalMemoryHandleDesc externalMemoryHandleDesc = {};
    externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    externalMemoryHandleDesc.handle.fd = fd;
    externalMemoryHandleDesc.size = size;

    if (cudaImportExternalMemory(&m_external_density, &externalMemoryHandleDesc) != cudaSuccess) {
        throw std::runtime_error("Failed to import external memory to CUDA");
    }

    cudaExternalMemoryBufferDesc bufferDesc = {};
    bufferDesc.offset = 0;
    bufferDesc.size = size;
    bufferDesc.flags = 0;

    void* mapped_ptr = nullptr;
    if (cudaExternalMemoryGetMappedBuffer(&mapped_ptr, m_external_density, &bufferDesc) != cudaSuccess) {
        throw std::runtime_error("Failed to map external memory to CUDA pointer");
    }

    // Replace d_rho with a non-owning buffer wrapping the mapped pointer
    d_rho = cuda_buffer<float>(static_cast<float*>(mapped_ptr), num_cells, false);
}

void lbm_solver::register_external_velocity(int fd, size_t size) {
    cudaExternalMemoryHandleDesc externalMemoryHandleDesc = {};
    externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    externalMemoryHandleDesc.handle.fd = fd;
    externalMemoryHandleDesc.size = size;

    if (cudaImportExternalMemory(&m_external_velocity, &externalMemoryHandleDesc) != cudaSuccess) {
        throw std::runtime_error("Failed to import external velocity memory to CUDA");
    }

    cudaExternalMemoryBufferDesc bufferDesc = {};
    bufferDesc.offset = 0;
    bufferDesc.size = size;
    bufferDesc.flags = 0;

    void* mapped_ptr = nullptr;
    if (cudaExternalMemoryGetMappedBuffer(&mapped_ptr, m_external_velocity, &bufferDesc) != cudaSuccess) {
        throw std::runtime_error("Failed to map external velocity memory to CUDA pointer");
    }

    // Replace d_u with a non-owning buffer wrapping the mapped pointer
    d_u = cuda_buffer<float2>(static_cast<float2*>(mapped_ptr), num_cells, false);
}

__global__ void k_compute_curl(const float2* __restrict__ u, float* __restrict__ curl, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x >= width || y >= height) return;

    // Simple finite differences
    // Curl = du_y/dx - du_x/dy
    
    // du_y/dx
    int x_plus = (x + 1 < width) ? (x + 1) : x;
    int x_minus = (x - 1 >= 0) ? (x - 1) : x;
    float uy_plus = u[y * width + x_plus].y;
    float uy_minus = u[y * width + x_minus].y;
    float duy_dx = (uy_plus - uy_minus) * 0.5f;

    // du_x/dy
    int y_plus = (y + 1 < height) ? (y + 1) : y;
    int y_minus = (y - 1 >= 0) ? (y - 1) : y;
    float ux_plus = u[y_plus * width + x].x;
    float ux_minus = u[y_minus * width + x].x;
    float dux_dy = (ux_plus - ux_minus) * 0.5f;

    curl[idx] = duy_dx - dux_dy;
}

void lbm_solver::compute_curl() {
    k_compute_curl<<<grid_size, block_size>>>(d_u.get_data(), d_curl.get_data(), width, height);
    cudaDeviceSynchronize();
}

void lbm_solver::register_external_curl(int fd, size_t size) {
    cudaExternalMemoryHandleDesc externalMemoryHandleDesc = {};
    externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    externalMemoryHandleDesc.handle.fd = fd;
    externalMemoryHandleDesc.size = size;

    if (cudaImportExternalMemory(&m_external_curl, &externalMemoryHandleDesc) != cudaSuccess) {
        throw std::runtime_error("Failed to import external curl memory to CUDA");
    }

    cudaExternalMemoryBufferDesc bufferDesc = {};
    bufferDesc.offset = 0;
    bufferDesc.size = size;
    bufferDesc.flags = 0;

    void* mapped_ptr = nullptr;
    if (cudaExternalMemoryGetMappedBuffer(&mapped_ptr, m_external_curl, &bufferDesc) != cudaSuccess) {
        throw std::runtime_error("Failed to map external curl memory to CUDA pointer");
    }

    // Replace d_curl with a non-owning buffer wrapping the mapped pointer
    d_curl = cuda_buffer<float>(static_cast<float*>(mapped_ptr), num_cells, false);
}

void lbm_solver::register_external_solid(int fd, size_t size) {
    cudaExternalMemoryHandleDesc externalMemoryHandleDesc = {};
    externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    externalMemoryHandleDesc.handle.fd = fd;
    externalMemoryHandleDesc.size = size;

    if (cudaImportExternalMemory(&m_external_solid, &externalMemoryHandleDesc) != cudaSuccess) {
        throw std::runtime_error("Failed to import external solid memory to CUDA");
    }

    cudaExternalMemoryBufferDesc bufferDesc = {};
    bufferDesc.offset = 0;
    bufferDesc.size = size;
    bufferDesc.flags = 0;

    void* mapped_ptr = nullptr;
    if (cudaExternalMemoryGetMappedBuffer(&mapped_ptr, m_external_solid, &bufferDesc) != cudaSuccess) {
        throw std::runtime_error("Failed to map external solid memory to CUDA pointer");
    }

    // Replace d_solid with a non-owning buffer wrapping the mapped pointer
    d_solid = cuda_buffer<unsigned char>(static_cast<unsigned char*>(mapped_ptr), num_cells, false);
}
