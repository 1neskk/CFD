#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "lbm_solver.cuh"

using namespace cuda_math;

// D3Q19 Constants
__constant__ float w[19] = {
    1.0f / 3.0f,  // 0: Center
    1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f,
    1.0f / 18.0f, 1.0f / 18.0f,                              // 1-6: Axial
    1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f,  // 7-10: XY plane
    1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f,  // 11-14: XZ plane
    1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f   // 15-18: YZ plane
};

__constant__ int cx[19] = {0,  1, -1, 0, 0,  0, 0, 1, -1, 1,
                           -1, 1, -1, 1, -1, 0, 0, 0, 0};
__constant__ int cy[19] = {0, 0, 0, 1, -1, 0, 0,  1, -1, -1,
                           1, 0, 0, 0, 0,  1, -1, 1, -1};
__constant__ int cz[19] = {0, 0, 0,  0,  0, 1, -1, 0,  0, 0,
                           0, 1, -1, -1, 1, 1, -1, -1, 1};

// Inverse directions for bounce-back
__constant__ int inv_dir[19] = {0, 2,  1,  4,  3,  6,  5,  8,  7, 10,
                                9, 12, 11, 14, 13, 16, 15, 18, 17

};

__constant__ int reflect_y[19] = {0, 1,  2,  4,  3,  5,  6,  9,  10, 7,
                                  8, 11, 12, 13, 14, 16, 15, 18, 17};

__constant__ int reflect_z[19] = {0,  1,  2,  3,  4,  6,  5,  7,  8, 9,
                                  10, 13, 14, 11, 12, 17, 18, 15, 16};

__global__ void k_init(float* f, float* rho, vec4* u, int width, int height,
                       int depth) {
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

    // Equilibrium
    float u_sq = length_sq(vec3(u_x, u_y, u_z));
    for (int k = 0; k < 19; k++) {
        float cu = cx[k] * u_x + cy[k] * u_y + cz[k] * u_z;
        float f_eq =
            w[k] * r * (1.0f + 3.0f * cu + 4.5f * cu * cu - 1.5f * u_sq);
        f[k * width * height * depth + idx] = f_eq;
    }

    rho[idx] = r;
    u[idx] = vec4(u_x, u_y, u_z, 0.0f);
}

// Deprecated kernel (might delete soon)
#if 0
__global__ void k_stream_collide(
    const float* __restrict__ f_in, float* __restrict__ f_out, 
    float* __restrict__ rho_out, vec4* __restrict__ u_out, 
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
    float f[19];
    bool is_solid = (solid[idx] > 0);

    if (is_solid) {
        rho_out[idx] = 0.0f;
        u_out[idx] = vec4(0.0f);

        for (int k = 0; k < 19; k++) {
            f_out[k * num_cells + idx] = f_in[k * num_cells + idx];
        }
        return; 
    }

    // Fluid cell processing - Streaming
    float rho = 0.0f;
    float ux = 0.0f;
    float uy = 0.0f;
    float uz = 0.0f;

    for (int k = 0; k < 19; k++) {
        int nx = x - cx[k];
        int ny = y - cy[k];
        int nz = z - cz[k];
        
        // Periodic wrap
        ny = (ny + height) % height;
        nz = (nz + depth) % depth;

        // Inlet/Outlet for X
        if (nx < 0) {
            // Inlet: Equilibrium with inlet_velocity
            float r = 1.0f;
            float u_in_x = inlet_velocity;
            float u_in_y = 0.0f;
            float u_in_z = 0.0f;
            float u_sq = length_sq(vec3(u_in_x, u_in_y, u_in_z));
            float cu = cx[k] * u_in_x + cy[k] * u_in_y + cz[k] * u_in_z;
            f[k] = w[k] * r * (1.0f + 3.0f * cu + 4.5f * cu * cu - 1.5f * u_sq);
        } else if (nx >= width) {
            // Outlet: Copy from right ghost cell
            int safe_nx = width - 1;
            int n_idx = nz * width * height + ny * width + safe_nx;
            f[k] = f_in[k * num_cells + n_idx];
        } else {
            int n_idx = nz * width * height + ny * width + nx;
            if (solid[n_idx] > 0) {
                int inv_k = inv_dir[k];
                f[k] = f_in[inv_k * num_cells + idx];
            } else {
                f[k] = f_in[k * num_cells + n_idx];
            }
        }

        rho += f[k];
        ux += f[k] * cx[k];
        uy += f[k] * cy[k];
        uz += f[k] * cz[k];
    }

    // 2. Macroscopic moments
    if (rho > 0.0f) {
        float inv_rho = 1.0f / rho;
        ux *= inv_rho;
        uy *= inv_rho;
        uz *= inv_rho;
    }

    rho_out[idx] = rho;
    u_out[idx] = vec4(ux, uy, uz, 0.0f);

    // 3. Cumulant Collision
    float omega = 1.0f / tau;
    float omega_high = 1.0f;

    // Pre-compute powers of velocity
    float ux2 = ux * ux;
    float uy2 = uy * uy;
    float uz2 = uz * uz;
    float uxuy = ux * uy;
    float uyuz = uy * uz;
    float uzux = uz * ux;

    float f_axis_x = f[1] + f[2];
    float f_axis_y = f[3] + f[4];
    float f_axis_z = f[5] + f[6];
    
    float f_plane_xy = f[7] + f[8] + f[9] + f[10];
    float f_plane_xz = f[11] + f[12] + f[13] + f[14];
    float f_plane_yz = f[15] + f[16] + f[17] + f[18];

    float m200 = f[1] + f[2] + f_plane_xy + f_plane_xz;
    float m020 = f[3] + f[4] + f_plane_xy + f_plane_yz;
    float m002 = f[5] + f[6] + f_plane_xz + f_plane_yz;
    
    float m110 = (f[7] + f[10]) - (f[8] + f[9]);
    float m101 = (f[11] + f[14]) - (f[12] + f[13]);
    float m011 = (f[15] + f[18]) - (f[16] + f[17]);

    float tr = m200 + m020 + m002;

    float k200 = m200 - rho * ux2;
    float k020 = m020 - rho * uy2;
    float k002 = m002 - rho * uz2;
    float k110 = m110 - rho * uxuy;
    float k101 = m101 - rho * uzux;
    float k011 = m011 - rho * uyuz;

    float k_xx_min_yy = k200 - k020;
    float k_xx_min_zz = k200 - k002; // or use 3*k200 - tr ?
    
    float one_third_rho = rho * (1.0f/3.0f);
    
    k200 = k200 - omega * (k200 - one_third_rho);
    k020 = k020 - omega * (k020 - one_third_rho);
    k002 = k002 - omega * (k002 - one_third_rho);
    k110 = k110 * (1.0f - omega);
    k011 = k011 * (1.0f - omega);
    k101 = k101 * (1.0f - omega);

    m200 = k200 + rho * ux2;
    m020 = k020 + rho * uy2;
    m002 = k002 + rho * uz2;
    m110 = k110 + rho * uxuy;
    m011 = k011 + rho * uyuz;
    m101 = k101 + rho * uzux;

    float P_neq_xx = k200 - one_third_rho;
    float P_neq_yy = k020 - one_third_rho;
    float P_neq_zz = k002 - one_third_rho;
    float P_neq_xy = k110;
    float P_neq_yz = k011;
    float P_neq_zx = k101;

    // 4. Reconstruction
    float u_sq = length_sq(vec3(ux, uy, uz));
    
    for (int k = 0; k < 19; k++) {
        float cu = cx[k] * ux + cy[k] * uy + cz[k] * uz;
        
        float f_eq = w[k] * rho * (1.0f + 3.0f * cu + 4.5f * cu * cu - 1.5f * u_sq);

        float Q_xx = (float)(cx[k] * cx[k]) - 1.0f/3.0f;
        float Q_yy = (float)(cy[k] * cy[k]) - 1.0f/3.0f;
        float Q_zz = (float)(cz[k] * cz[k]) - 1.0f/3.0f;
        float Q_xy = (float)(cx[k] * cy[k]);
        float Q_yz = (float)(cy[k] * cz[k]);
        float Q_zx = (float)(cz[k] * cx[k]);

        float f_neq = 4.5f * w[k] * (
            Q_xx * P_neq_xx + Q_yy * P_neq_yy + Q_zz * P_neq_zz +
            2.0f * (Q_xy * P_neq_xy + Q_yz * P_neq_yz + Q_zx * P_neq_zx)
        );

        f_out[k * num_cells + idx] = f_eq + f_neq;
    }
}
#endif

__global__ void k_stream_collide_cumulant(
    const float* __restrict__ f_in, float* __restrict__ f_out,
    float* __restrict__ rho_out, vec4* __restrict__ u_out,
    const unsigned char* __restrict__ solid, int width, int height, int depth,
    float inlet_velocity, float omega_shear, float omega_bulk) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= width || y >= height || z >= depth) return;

    int idx = z * width * height + y * width + x;
    int num_cells = width * height * depth;

    // 1. Stream & Bounce-back
    float f[19];
    bool is_solid = (solid[idx] > 0);

    if (is_solid) {
        rho_out[idx] = 0.0f;
        u_out[idx] = vec4(0.0f);
        for (int k = 0; k < 19; k++) {
            f_out[k * num_cells + idx] = f_in[k * num_cells + idx];
        }
        return;
    }

    float rho = 0.0f;
    float ux = 0.0f;
    float uy = 0.0f;
    float uz = 0.0f;

    for (int k = 0; k < 19; k++) {
        int nx = x - cx[k];
        int ny = y - cy[k];
        int nz = z - cz[k];

        // Periodic wrap - REMOVED for Wind Tunnel
        // ny = (ny + height) % height;
        // nz = (nz + depth) % depth;

        if (nx < 0) {
            // Inlet: Equilibrium with inlet_velocity
            float r = 1.0f;
            float u_in_x = inlet_velocity;
            float u_in_y = 0.0f;
            float u_in_z = 0.0f;
            float u_sq = u_in_x * u_in_x + u_in_y * u_in_y + u_in_z * u_in_z;
            float cu = cx[k] * u_in_x + cy[k] * u_in_y + cz[k] * u_in_z;
            f[k] = w[k] * r * (1.0f + 3.0f * cu + 4.5f * cu * cu - 1.5f * u_sq);
        } else if (nx >= width) {
            // Outlet: Copy from right ghost cell (simple extrapolation)
            int safe_nx = width - 1;
            int n_idx = nz * width * height + ny * width + safe_nx;
            f[k] = f_in[k * num_cells + n_idx];
        } else if (ny < 0) {
            // Floor (Y=0) - Should be solid, but if not, bounce-back
            int inv_k = inv_dir[k];
            f[k] = f_in[inv_k * num_cells + idx];
        } else if (ny >= height) {
            // Ceiling (Y=H-1) - Free Slip
            int ref_k = reflect_y[k];
            f[k] = f_in[ref_k * num_cells + idx];
        } else if (nz < 0 || nz >= depth) {
            // Side Walls (Z=0, Z=D-1) - Free Slip
            int ref_k = reflect_z[k];
            f[k] = f_in[ref_k * num_cells + idx];
        } else {
            int n_idx = nz * width * height + ny * width + nx;
            if (solid[n_idx] > 0) {
                // Bounce-back
                int inv_k = inv_dir[k];
                f[k] = f_in[inv_k * num_cells + idx];
            } else {
                f[k] = f_in[k * num_cells + n_idx];
            }
        }

        rho += f[k];
        ux += f[k] * cx[k];
        uy += f[k] * cy[k];
        uz += f[k] * cz[k];
    }

    // 2. Macroscopic moments
    if (rho > 1e-9f) {
        float inv_rho = 1.0f / rho;
        ux *= inv_rho;
        uy *= inv_rho;
        uz *= inv_rho;
    } else {
        rho = 1.0f;  // Safety
    }

    rho_out[idx] = rho;
    u_out[idx] = vec4(ux, uy, uz, 0.0f);

    // Pre-compute raw second moments
    float m_xx = 0.0f, m_yy = 0.0f, m_zz = 0.0f;
    float m_xy = 0.0f, m_yz = 0.0f, m_zx = 0.0f;

    // Unroll loop for efficiency or just loop
    for (int k = 0; k < 19; k++) {
        float fk = f[k];
        m_xx += fk * cx[k] * cx[k];
        m_yy += fk * cy[k] * cy[k];
        m_zz += fk * cz[k] * cz[k];
        m_xy += fk * cx[k] * cy[k];
        m_yz += fk * cy[k] * cz[k];
        m_zx += fk * cz[k] * cx[k];
    }

    float P_xx_neq = m_xx - rho * ux * ux - rho * (1.0f / 3.0f);
    float P_yy_neq = m_yy - rho * uy * uy - rho * (1.0f / 3.0f);
    float P_zz_neq = m_zz - rho * uz * uz - rho * (1.0f / 3.0f);
    float P_xy_neq = m_xy - rho * ux * uy;
    float P_yz_neq = m_yz - rho * uy * uz;
    float P_zx_neq = m_zx - rho * uz * ux;

    float trace = P_xx_neq + P_yy_neq + P_zz_neq;
    float trace_relaxed = trace * (1.0f - omega_bulk);

    // Deviatoric parts relate to shear viscosity
    // D_ab = P_ab - 1/3 * trace * delta_ab
    // We want to relax D_ab by omega_shear
    // P_ab_new = D_ab_relaxed + 1/3 * trace_relaxed * delta_ab
    //          = D_ab * (1 - w_s) + 1/3 * trace * (1 - w_b) * delta_ab
    //          = (P_ab - 1/3 * trace * delta_ab) * (1 - w_s) + ...

    // Optimized:
    // P_xx_new = (P_xx_neq - trace/3) * (1 - w_s) + (trace/3) * (1 - w_b)
    //          = P_xx_neq * (1 - w_s) + trace/3 * (w_s - w_b)

    float third_trace = trace * (1.0f / 3.0f);
    float correction = third_trace * (omega_shear - omega_bulk);

    float P_xx_new = P_xx_neq * (1.0f - omega_shear) + correction;
    float P_yy_new = P_yy_neq * (1.0f - omega_shear) + correction;
    float P_zz_new = P_zz_neq * (1.0f - omega_shear) + correction;

    float P_xy_new = P_xy_neq * (1.0f - omega_shear);
    float P_yz_new = P_yz_neq * (1.0f - omega_shear);
    float P_zx_new = P_zx_neq * (1.0f - omega_shear);

    // 5. Reconstruction
    // f = f_eq + f_neq
    // f_neq = w_k * 4.5 * (Q_ab : P_ab_new)
    // where Q_ab = c_a c_b - 1/3 delta_ab

    float u_sq = ux * ux + uy * uy + uz * uz;

    for (int k = 0; k < 19; k++) {
        // Equilibrium
        float cu = cx[k] * ux + cy[k] * uy + cz[k] * uz;
        float f_eq =
            w[k] * rho * (1.0f + 3.0f * cu + 4.5f * cu * cu - 1.5f * u_sq);

        // Non-Equilibrium
        float Q_xx = (float)(cx[k] * cx[k]) - 1.0f / 3.0f;
        float Q_yy = (float)(cy[k] * cy[k]) - 1.0f / 3.0f;
        float Q_zz = (float)(cz[k] * cz[k]) - 1.0f / 3.0f;
        float Q_xy = (float)(cx[k] * cy[k]);
        float Q_yz = (float)(cy[k] * cz[k]);
        float Q_zx = (float)(cz[k] * cx[k]);

        float f_neq =
            4.5f * w[k] *
            (Q_xx * P_xx_new + Q_yy * P_yy_new + Q_zz * P_zz_new +
             2.0f * (Q_xy * P_xy_new + Q_yz * P_yz_new + Q_zx * P_zx_new));

        f_out[k * num_cells + idx] = f_eq + f_neq;
    }
}

lbm_solver::lbm_solver(int width, int height, int depth)
    : m_width(width), m_height(height), m_depth(depth) {
    m_num_cells = width * height * depth;

    d_f = cuda_buffer<float>(19 * m_num_cells);
    d_f_new = cuda_buffer<float>(19 * m_num_cells);
    d_rho = cuda_buffer<float>(m_num_cells);
    d_u = cuda_buffer<vec4>(m_num_cells);
    d_curl = cuda_buffer<float>(m_num_cells);
    d_solid = cuda_buffer<unsigned char>(m_num_cells);

    d_solid.memset(0);

    cudaDeviceProp prop;
    int min_grid_size, optimal_block_size;
#ifdef _DEBUG
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
        &min_grid_size, &optimal_block_size, k_stream_collide_cumulant, 0, 0));

    LOG_INFO("Optimal block size: {}", optimal_block_size);
#else
    cudaGetDeviceProperties(&prop, 0);
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &optimal_block_size,
                                       k_stream_collide_cumulant, 0, 0);
#endif
    int bx = 32;
    int remaining = optimal_block_size / bx;
    int by = floor(sqrt(static_cast<float>(remaining)));
    int bz = remaining / by;

    m_block_size = dim3(bx, by, bz);
    m_grid_size = dim3((m_width + m_block_size.x - 1) / m_block_size.x,
                       (m_height + m_block_size.y - 1) / m_block_size.y,
                       (m_depth + m_block_size.z - 1) / m_block_size.z);

#ifdef _DEBUG
    if (m_block_size.x * m_block_size.y >
        static_cast<unsigned int>(optimal_block_size))
        LOG_ERROR("Block size exceeds optimal size; ajust accordingly.");

    LOG_INFO("Block size: ({}, {}, {})", m_block_size.x, m_block_size.y,
             m_block_size.z);
    LOG_INFO("Grid size: ({}, {}, {})", m_grid_size.x, m_grid_size.y,
             m_grid_size.z);
    LOG_INFO("Threads per block: {}", m_block_size.x * m_block_size.y);
#endif

    m_voxelizer = new MeshVoxelizer();
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

    delete m_voxelizer;
}

void lbm_solver::init() {
    k_init<<<m_grid_size, m_block_size>>>(d_f.get_data(), d_rho.get_data(),
                                          d_u.get_data(), m_width, m_height,
                                          m_depth);
}

void lbm_solver::step() {
    // k_stream_collide<<<m_grid_size, m_block_size>>>(d_f.get_data(),
    // d_f_new.get_data(), d_rho.get_data(), d_u.get_data(), d_solid.get_data(),
    // m_width, m_height, m_depth, m_settings.tau, m_settings.inlet_velocity);

    k_stream_collide_cumulant<<<m_grid_size, m_block_size>>>(
        d_f.get_data(), d_f_new.get_data(), d_rho.get_data(), d_u.get_data(),
        d_solid.get_data(), m_width, m_height, m_depth,
        m_settings.inlet_velocity, m_settings.omega_shear,
        m_settings.omega_bulk);

    compute_curl();

    std::swap(d_f, d_f_new);
#ifdef _DEBUG
    cudaDeviceSynchronize();
#endif
}

void lbm_solver::reset() { init(); }

__global__ void k_add_solid_rect(unsigned char* solid, int width, int height,
                                 int depth, int rx, int ry, int rz, int rw,
                                 int rh, int rd) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= width || y >= height || z >= depth) return;

    int idx = z * width * height + y * width + x;

    if (x >= rx && x < rx + rw && y >= ry && y < ry + rh && z >= rz &&
        z < rz + rd) {
        solid[idx] = 255;
    }
}

void lbm_solver::add_solid(const Rect& rect) {
    k_add_solid_rect<<<m_grid_size, m_block_size>>>(
        d_solid.get_data(), m_width, m_height, m_depth, rect.x, rect.y, rect.z,
        rect.w, rect.h, rect.d);
}

void lbm_solver::load_mesh(const std::string& path) {
    if (!m_voxelizer->load_obj(path)) {
        LOG_ERROR("Failed to load mesh: {}", path);
    }

    std::vector<unsigned char> host_solid(m_num_cells, 0);

    m_voxelizer->rotate_y(-90.0f);

    glm::vec3 mesh_size =
        m_voxelizer->get_max_bound() - m_voxelizer->get_min_bound();

    float scale_x = (static_cast<float>(m_width) * 0.8f) / mesh_size.x;
    float scale_y = (static_cast<float>(m_height) * 0.8f) / mesh_size.y;
    float scale_z = (static_cast<float>(m_depth) * 0.8f) / mesh_size.z;
    LOG_INFO("Scale: [{}, {}, {}]", scale_x, scale_y, scale_z);

    float scale = std::min({scale_x, scale_y, scale_z});

    LOG_INFO(
        "Auto-scaling mesh. Mesh Size: [{}, {}, {}], Grid Size: [{}, {}, {}], "
        "Scale: {}",
        mesh_size.x, mesh_size.y, mesh_size.z, m_width, m_height, m_depth,
        scale);

    m_voxelizer->voxelize(host_solid.data(), m_width, m_height, m_depth, scale);

    // y = 0 is solid
    for (int z = 0; z < m_depth; z++) {
        for (int x = 0; x < m_width; x++) {
            host_solid[z * m_width * m_height + 0 * m_width + x] = 255;
        }
    }

    d_solid.copy_from_host(host_solid.data(), m_num_cells);
}

void lbm_solver::register_external_density(int fd, size_t size) {
    cudaExternalMemoryHandleDesc externalMemoryHandleDesc = {};
    externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    externalMemoryHandleDesc.handle.fd = fd;
    externalMemoryHandleDesc.size = size;

    if (cudaImportExternalMemory(&m_external_density,
                                 &externalMemoryHandleDesc) != cudaSuccess) {
        throw std::runtime_error("Failed to import external memory to CUDA");
    }

    cudaExternalMemoryBufferDesc bufferDesc = {};
    bufferDesc.offset = 0;
    bufferDesc.size = size;
    bufferDesc.flags = 0;

    void* mapped_ptr = nullptr;
    if (cudaExternalMemoryGetMappedBuffer(&mapped_ptr, m_external_density,
                                          &bufferDesc) != cudaSuccess) {
        throw std::runtime_error(
            "Failed to map external memory to CUDA pointer");
    }

    d_rho =
        cuda_buffer<float>(static_cast<float*>(mapped_ptr), m_num_cells, false);
}

void lbm_solver::register_external_velocity(int fd, size_t size) {
    cudaExternalMemoryHandleDesc externalMemoryHandleDesc = {};
    externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    externalMemoryHandleDesc.handle.fd = fd;
    externalMemoryHandleDesc.size = size;

    if (cudaImportExternalMemory(&m_external_velocity,
                                 &externalMemoryHandleDesc) != cudaSuccess) {
        throw std::runtime_error(
            "Failed to import external velocity memory to CUDA");
    }

    cudaExternalMemoryBufferDesc bufferDesc = {};
    bufferDesc.offset = 0;
    bufferDesc.size = size;
    bufferDesc.flags = 0;

    void* mapped_ptr = nullptr;
    if (cudaExternalMemoryGetMappedBuffer(&mapped_ptr, m_external_velocity,
                                          &bufferDesc) != cudaSuccess) {
        throw std::runtime_error(
            "Failed to map external velocity memory to CUDA pointer");
    }

    d_u = cuda_buffer<vec4>(static_cast<vec4*>(mapped_ptr), m_num_cells, false);
}

__global__ void k_compute_curl(const vec4* __restrict__ u,
                               float* __restrict__ curl, int width, int height,
                               int depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= width || y >= height || z >= depth) return;

    int idx = z * width * height + y * width + x;

    // Helper to get velocity safely
    auto get_u = [&](int ix, int iy, int iz) -> vec4 {
        ix = max(0, min(ix, width - 1));
        iy = max(0, min(iy, height - 1));
        iz = max(0, min(iz, depth - 1));
        return u[iz * width * height + iy * width + ix];
    };

    vec4 u_c = get_u(x, y, z);

    // Central differences
    vec4 u_xp = get_u(x + 1, y, z);
    vec4 u_xm = get_u(x - 1, y, z);
    vec4 u_yp = get_u(x, y + 1, z);
    vec4 u_ym = get_u(x, y - 1, z);
    vec4 u_zp = get_u(x, y, z + 1);
    vec4 u_zm = get_u(x, y, z - 1);

    // Partial derivatives
    float dv_dx = (u_xp.y - u_xm.y) * 0.5f;
    float dw_dx = (u_xp.z - u_xm.z) * 0.5f;

    float du_dy = (u_yp.x - u_ym.x) * 0.5f;
    float dw_dy = (u_yp.z - u_ym.z) * 0.5f;

    float du_dz = (u_zp.x - u_zm.x) * 0.5f;
    float dv_dz = (u_zp.y - u_zm.y) * 0.5f;

    float curl_x = dw_dy - dv_dz;
    float curl_y = du_dz - dw_dx;
    float curl_z = dv_dx - du_dy;

    curl[idx] = length(vec3(curl_x, curl_y, curl_z));
}

void lbm_solver::compute_curl() {
    k_compute_curl<<<m_grid_size, m_block_size>>>(
        d_u.get_data(), d_curl.get_data(), m_width, m_height, m_depth);
}

void lbm_solver::register_external_curl(int fd, size_t size) {
    cudaExternalMemoryHandleDesc externalMemoryHandleDesc = {};
    externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    externalMemoryHandleDesc.handle.fd = fd;
    externalMemoryHandleDesc.size = size;

    if (cudaImportExternalMemory(&m_external_curl, &externalMemoryHandleDesc) !=
        cudaSuccess) {
        throw std::runtime_error(
            "Failed to import external curl memory to CUDA");
    }

    cudaExternalMemoryBufferDesc bufferDesc = {};
    bufferDesc.offset = 0;
    bufferDesc.size = size;
    bufferDesc.flags = 0;

    void* mapped_ptr = nullptr;
    if (cudaExternalMemoryGetMappedBuffer(&mapped_ptr, m_external_curl,
                                          &bufferDesc) != cudaSuccess) {
        throw std::runtime_error(
            "Failed to map external curl memory to CUDA pointer");
    }

    d_curl =
        cuda_buffer<float>(static_cast<float*>(mapped_ptr), m_num_cells, false);
}

void lbm_solver::register_external_solid(int fd, size_t size) {
    cudaExternalMemoryHandleDesc externalMemoryHandleDesc = {};
    externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    externalMemoryHandleDesc.handle.fd = fd;
    externalMemoryHandleDesc.size = size;

    if (cudaImportExternalMemory(&m_external_solid,
                                 &externalMemoryHandleDesc) != cudaSuccess) {
        throw std::runtime_error(
            "Failed to import external solid memory to CUDA");
    }

    cudaExternalMemoryBufferDesc bufferDesc = {};
    bufferDesc.offset = 0;
    bufferDesc.size = size;
    bufferDesc.flags = 0;

    void* mapped_ptr = nullptr;
    if (cudaExternalMemoryGetMappedBuffer(&mapped_ptr, m_external_solid,
                                          &bufferDesc) != cudaSuccess) {
        throw std::runtime_error(
            "Failed to map external solid memory to CUDA pointer");
    }

    d_solid = cuda_buffer<unsigned char>(
        static_cast<unsigned char*>(mapped_ptr), m_num_cells, false);
}
