#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>

#include "logger.h"

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            LOG_ERROR("CUDA error: {} at {}:{}", cudaGetErrorString(err), __FILE__, __LINE__); \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

template <typename T>
class cuda_buffer {
   public:
    cuda_buffer() = default;
    explicit cuda_buffer(size_t size) : m_size(size) {
        if (size > 0) {
            CUDA_CHECK(cudaMalloc(&m_data, size * sizeof(T)));
        }
    }

    cuda_buffer(T* ptr, size_t size, bool owns_memory = false) 
        : m_data(ptr), m_size(size), m_owns_memory(owns_memory) {}

    ~cuda_buffer() { release(); }

    cuda_buffer(const cuda_buffer &) = default;
    cuda_buffer &operator=(const cuda_buffer &) = delete;

    cuda_buffer(cuda_buffer &&other) noexcept
        : m_data(other.m_data), m_size(other.m_size), m_owns_memory(other.m_owns_memory) {
        other.m_data = nullptr;
        other.m_size = 0;
        other.m_owns_memory = true;
    }

    cuda_buffer &operator=(cuda_buffer &&other) noexcept {
        if (this != &other) {
            release();
            m_data = other.m_data;
            m_size = other.m_size;
            m_owns_memory = other.m_owns_memory;
            other.m_data = nullptr;
            other.m_size = 0;
            other.m_owns_memory = true;
        }
        return *this;
    }

    void release() {
        if (m_data && m_owns_memory) {
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaFree(m_data));
        }
        m_data = nullptr;
        m_size = 0;
        m_owns_memory = true;
    }
    
    void memset(int value) {
        if (m_data) {
            CUDA_CHECK(cudaMemset(m_data, value, m_size * sizeof(T)));
        }
    }

    void copy_from_host(const T *host_data, size_t size) {
        if (m_data && size <= m_size) {
            CUDA_CHECK(cudaMemcpy(m_data, host_data, size * sizeof(T),
            cudaMemcpyHostToDevice));
        }
    }

    void copy_to_host(T *host_data, size_t size) {
        if (m_data && size <= m_size) {
            CUDA_CHECK(cudaMemcpy(host_data, m_data, size * sizeof(T),
                                  cudaMemcpyDeviceToHost));
        }
    }

    void copy_from_device(const T* device_ptr, size_t size) {
        if (m_data && size <= m_size) {
            CUDA_CHECK(cudaMemcpy(m_data, device_ptr, size * sizeof(T),
                                  cudaMemcpyDeviceToDevice));
        }
    }

    void copy_to_device(T* device_ptr, size_t size) const {
        if (m_data && size <= m_size) {
            CUDA_CHECK(cudaMemcpy(device_ptr, m_data, size * sizeof(T),
                                  cudaMemcpyDeviceToDevice));
        }
    }

    __host__ T *get_data() {
        if (!m_data) {
            LOG_ERROR("Trying to access null device pointer!");
        }

        return m_data;
    }

    __host__ [[nodiscard]] const T *get_data() const {
        if (!m_data) {
            LOG_ERROR("Trying to access null device pointer!");
        }

        return m_data;
    }

    void set_size(const size_t size) { m_size = size; }

private:
    T *m_data = nullptr;
    size_t m_size = 0;
    bool m_owns_memory = true;
};