#pragma once

#include <vulkan/vulkan.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <stdexcept>

class vulkan_cuda_buffer {
public:
    vulkan_cuda_buffer(VkPhysicalDevice physicalDevice, VkDevice device, size_t size) 
        : m_device(device), m_size(size) {
        
        // 1. Create Vulkan Buffer
        VkBufferCreateInfo bufferInfo = {};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        // Add external memory handle type
        VkExternalMemoryBufferCreateInfo externalMemoryBufferInfo = {};
        externalMemoryBufferInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
        externalMemoryBufferInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        bufferInfo.pNext = &externalMemoryBufferInfo;

        if (vkCreateBuffer(device, &bufferInfo, nullptr, &m_buffer) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create Vulkan buffer");
        }

        // 2. Allocate Memory
        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, m_buffer, &memRequirements);

        VkExportMemoryAllocateInfo exportAllocInfo = {};
        exportAllocInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
        exportAllocInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

        VkMemoryAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = find_memory_type(physicalDevice, memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        allocInfo.pNext = &exportAllocInfo;

        if (vkAllocateMemory(device, &allocInfo, nullptr, &m_memory) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate Vulkan memory");
        }

        if (vkBindBufferMemory(device, m_buffer, m_memory, 0) != VK_SUCCESS) {
            throw std::runtime_error("Failed to bind Vulkan buffer memory");
        }

        // 3. Export Memory to FD
        int fd;
        VkMemoryGetFdInfoKHR memoryGetFdInfo = {};
        memoryGetFdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
        memoryGetFdInfo.memory = m_memory;
        memoryGetFdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

        auto vkGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR");
        if (!vkGetMemoryFdKHR) {
            throw std::runtime_error("Failed to load vkGetMemoryFdKHR");
        }

        if (vkGetMemoryFdKHR(device, &memoryGetFdInfo, &fd) != VK_SUCCESS) {
            throw std::runtime_error("Failed to get memory FD");
        }

        // 4. Import to CUDA
        cudaExternalMemoryHandleDesc externalMemoryHandleDesc = {};
        externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
        externalMemoryHandleDesc.handle.fd = fd;
        externalMemoryHandleDesc.size = memRequirements.size;

        if (cudaImportExternalMemory(&m_external_memory, &externalMemoryHandleDesc) != cudaSuccess) {
            throw std::runtime_error("Failed to import external memory to CUDA");
        }

        // 5. Map to CUDA Pointer
        cudaExternalMemoryBufferDesc bufferDesc = {};
        bufferDesc.offset = 0;
        bufferDesc.size = size;
        bufferDesc.flags = 0;

        if (cudaExternalMemoryGetMappedBuffer(&m_cuda_ptr, m_external_memory, &bufferDesc) != cudaSuccess) {
            throw std::runtime_error("Failed to map external memory to CUDA pointer");
        }
    }

    ~vulkan_cuda_buffer() {
        if (m_cuda_ptr) {
            cudaFree(m_cuda_ptr);
        }
        if (m_external_memory) {
            cudaDestroyExternalMemory(m_external_memory);
        }
        if (m_buffer) {
            vkDestroyBuffer(m_device, m_buffer, nullptr);
        }
        if (m_memory) {
            vkFreeMemory(m_device, m_memory, nullptr);
        }
    }

    void *get_cuda_ptr() const { return m_cuda_ptr; }
    VkBuffer get_vk_buffer() const { return m_buffer; }
    size_t get_size() const { return m_size; }

private:
    uint32_t find_memory_type(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("Failed to find suitable memory type");
    }

    VkDevice m_device;
    size_t m_size;
    VkBuffer m_buffer = VK_NULL_HANDLE;
    VkDeviceMemory m_memory = VK_NULL_HANDLE;
    cudaExternalMemory_t m_external_memory = nullptr;
    void *m_cuda_ptr = nullptr;
};
