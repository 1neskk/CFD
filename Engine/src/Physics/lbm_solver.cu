#include "lbm_solver.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>

// D3Q19 Constants
__constant__ float w[19] = {
    1.0f/3.0f,  // 0: Center
    1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f, // 1-6: Axial
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, // 7-10: XY plane
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, // 11-14: XZ plane
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f  // 15-18: YZ plane
};

__constant__ int cx[19] = { 0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0 };
__constant__ int cy[19] = { 0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1 };
__constant__ int cz[19] = { 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1 };

// Inverse directions for bounce-back
__constant__ int inv_dir[19] = { 
    0, 
    2, 1, 
    4, 3, 
    6, 5, 
    8, 7, 
    10, 9, 
    12, 11, 
    14, 13, 
    16, 15, 
    18, 17 
};

__global__ void k_init(float* f, float* rho, float4* u, int width, int height, int depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= width || y >= height || z >= depth) return;

    int idx = z * width * height + y * width + x;

    // Initial conditions: rho = 1.0, u = 0
    float r = 1.0f;
    float u_x = 0.0f;
    float u_y = 0.0f;
    float u_z = 0.0f;

    // Compute equilibrium
    float u_sq = u_x * u_x + u_y * u_y + u_z * u_z;
    for (int k = 0; k < 19; k++) {
        float cu = cx[k] * u_x + cy[k] * u_y + cz[k] * u_z;
        float f_eq = w[k] * r * (1.0f + 3.0f * cu + 4.5f * cu * cu - 1.5f * u_sq);
        f[k * width * height * depth + idx] = f_eq;
    }

    rho[idx] = r;
    u[idx] = make_float4(u_x, u_y, u_z, 0.0f);
}

__global__ void k_stream_collide(
    const float* __restrict__ f_in, float* __restrict__ f_out, 
    float* __restrict__ rho_out, float4* __restrict__ u_out, 
    const unsigned char* __restrict__ solid,
    int width, int height, int depth, float tau, float inlet_velocity
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= width || y >= height || z >= depth) return;

    int idx = z * width * height + y * width + x;
    int num_cells = width * height * depth;

    // 1. Stream & Bounce-back
    float f_curr[19];
    bool is_solid = (solid[idx] > 0);

    if (is_solid) {
        rho_out[idx] = 0.0f;
        u_out[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        for (int k = 0; k < 19; k++) {
            f_out[k * num_cells + idx] = f_in[k * num_cells + idx];
        }
        return; 
    }

    // Fluid cell processing
    float rho = 0.0f;
    float ux = 0.0f;
    float uy = 0.0f;
    float uz = 0.0f;

    for (int k = 0; k < 19; k++) {
        int nx = x - cx[k];
        int ny = y - cy[k];
        int nz = z - cz[k];

        // Periodic boundaries for Y and Z (Wind tunnel walls are usually solid, but let's keep periodic for now or solid?)
        // Let's make Y and Z periodic for now to simulate infinite domain, or solid walls?
        // Task says "Wind Tunnel", usually walls are solid. But let's stick to simple periodic/inlet-outlet for now.
        // Actually, let's make Y and Z periodic for simplicity in this phase, or clamp.
        // Let's use periodic for Y and Z for now.
        
        if (ny < 0) ny += height;
        if (ny >= height) ny -= height;
        
        if (nz < 0) nz += depth;
        if (nz >= depth) nz -= depth;

        // Inlet/Outlet for X
        if (nx < 0) {
            // Inlet: Equilibrium with inlet_velocity
            float r = 1.0f;
            float u_in_x = inlet_velocity;
            float u_in_y = 0.0f;
            float u_in_z = 0.0f;
            float u_sq = u_in_x * u_in_x + u_in_y * u_in_y + u_in_z * u_in_z;
            float cu = cx[k] * u_in_x + cy[k] * u_in_y + cz[k] * u_in_z;
            f_curr[k] = w[k] * r * (1.0f + 3.0f * cu + 4.5f * cu * cu - 1.5f * u_sq);
        } else if (nx >= width) {
            // Outlet: Copy from right ghost cell (simple extrapolation)
            int safe_nx = width - 1;
            int n_idx = nz * width * height + ny * width + safe_nx;
            f_curr[k] = f_in[k * num_cells + n_idx];
        } else {
            int n_idx = nz * width * height + ny * width + nx;

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
        uz += f_curr[k] * cz[k];
    }

    // 2. Macroscopic moments
    if (rho > 0.0f) {
        ux /= rho;
        uy /= rho;
        uz /= rho;
    }

    rho_out[idx] = rho;
    u_out[idx] = make_float4(ux, uy, uz, 0.0f);

    // 3. Collide (BGK)
    float u_sq = ux * ux + uy * uy + uz * uz;
    float omega = 1.0f / tau;

    for (int k = 0; k < 19; k++) {
        float cu = cx[k] * ux + cy[k] * uy + cz[k] * uz;
        float f_eq = w[k] * rho * (1.0f + 3.0f * cu + 4.5f * cu * cu - 1.5f * u_sq);
        
        f_out[k * num_cells + idx] = f_curr[k] * (1.0f - omega) + f_eq * omega;
    }
}

lbm_solver::lbm_solver(int width, int height, int depth) : width(width), height(height), depth(depth) {
    num_cells = width * height * depth;
    
    d_f = cuda_buffer<float>(19 * num_cells);
    d_f_new = cuda_buffer<float>(19 * num_cells);
    d_rho = cuda_buffer<float>(num_cells);
    d_u = cuda_buffer<float4>(num_cells);
    d_curl = cuda_buffer<float>(num_cells);
    d_solid = cuda_buffer<unsigned char>(num_cells);

    // Zero out solid mask initially
    d_solid.memset(0);

    block_size = dim3(8, 8, 8); // 512 threads
    grid_size = dim3(
        (width + block_size.x - 1) / block_size.x, 
        (height + block_size.y - 1) / block_size.y,
        (depth + block_size.z - 1) / block_size.z
    );
}

lbm_solver::~lbm_solver() {
    d_f.release();
    d_f_new.release();
    d_rho.release();
    d_u.release();
    d_curl.release();
    d_solid.release();
    
    if (m_external_density) cudaDestroyExternalMemory(m_external_density);
    if (m_external_velocity) cudaDestroyExternalMemory(m_external_velocity);
    if (m_external_curl) cudaDestroyExternalMemory(m_external_curl);
    if (m_external_solid) cudaDestroyExternalMemory(m_external_solid);
}

void lbm_solver::init() {
    k_init<<<grid_size, block_size>>>(d_f.get_data(), d_rho.get_data(), d_u.get_data(), width, height, depth);
    cudaDeviceSynchronize();
}

void lbm_solver::step() {
    k_stream_collide<<<grid_size, block_size>>>(d_f.get_data(), d_f_new.get_data(), d_rho.get_data(), d_u.get_data(), d_solid.get_data(), width, height, depth, tau, inlet_velocity);
    cudaDeviceSynchronize();

    compute_curl();

    std::swap(d_f, d_f_new);
}

void lbm_solver::reset() {
    init();
    d_solid.memset(0);
}

__global__ void k_add_solid_rect(unsigned char* solid, int width, int height, int depth, int rx, int ry, int rz, int rw, int rh, int rd) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= width || y >= height || z >= depth) return;

    int idx = z * width * height + y * width + x;

    if (x >= rx && x < rx + rw && 
        y >= ry && y < ry + rh &&
        z >= rz && z < rz + rd) {
        solid[idx] = 255;
    }
}

void lbm_solver::add_solid(const Rect& rect) {
    k_add_solid_rect<<<grid_size, block_size>>>(d_solid.get_data(), width, height, depth, rect.x, rect.y, rect.z, rect.w, rect.h, rect.d);
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

    d_u = cuda_buffer<float4>(static_cast<float4*>(mapped_ptr), num_cells, false);
}

__global__ void k_compute_curl(const float4* __restrict__ u, float* __restrict__ curl, int width, int height, int depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= width || y >= height || z >= depth) return;

    int idx = z * width * height + y * width + x;

    // 3D Curl = (dw/dy - dv/dz, du/dz - dw/dx, dv/dx - du/dy)
    // For now, let's just compute magnitude of curl or a specific component?
    // The previous 2D code computed scalar curl (z-component).
    // Let's compute the magnitude of the curl vector.
    
    // Helper to get velocity safely
    auto get_u = [&](int ix, int iy, int iz) -> float4 {
        ix = max(0, min(ix, width - 1));
        iy = max(0, min(iy, height - 1));
        iz = max(0, min(iz, depth - 1));
        return u[iz * width * height + iy * width + ix];
    };

    float4 u_c = get_u(x, y, z);
    
    // Central differences
    float4 u_xp = get_u(x + 1, y, z);
    float4 u_xm = get_u(x - 1, y, z);
    float4 u_yp = get_u(x, y + 1, z);
    float4 u_ym = get_u(x, y - 1, z);
    float4 u_zp = get_u(x, y, z + 1);
    float4 u_zm = get_u(x, y, z - 1);

    // Partial derivatives
    // du/dx, du/dy, du/dz ...
    // u.x = u, u.y = v, u.z = w
    
    float dv_dx = (u_xp.y - u_xm.y) * 0.5f;
    float dw_dx = (u_xp.z - u_xm.z) * 0.5f;
    
    float du_dy = (u_yp.x - u_ym.x) * 0.5f;
    float dw_dy = (u_yp.z - u_ym.z) * 0.5f;
    
    float du_dz = (u_zp.x - u_zm.x) * 0.5f;
    float dv_dz = (u_zp.y - u_zm.y) * 0.5f;

    float curl_x = dw_dy - dv_dz;
    float curl_y = du_dz - dw_dx;
    float curl_z = dv_dx - du_dy;

    curl[idx] = sqrtf(curl_x * curl_x + curl_y * curl_y + curl_z * curl_z);
}

void lbm_solver::compute_curl() {
    k_compute_curl<<<grid_size, block_size>>>(d_u.get_data(), d_curl.get_data(), width, height, depth);
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

    d_solid = cuda_buffer<unsigned char>(static_cast<unsigned char*>(mapped_ptr), num_cells, false);
}
