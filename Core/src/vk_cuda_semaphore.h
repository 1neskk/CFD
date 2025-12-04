#pragma once

#include <vulkan/vulkan.h>
#include <cuda_runtime.h>
#include <iostream>

class cuda_vulkan_semaphore {
public:
    cuda_vulkan_semaphore(VkDevice device) : m_device(device) {
        VkExportSemaphoreCreateInfo exportInfo = {};
        exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
        exportInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;

        VkSemaphoreCreateInfo semaphoreInfo = {};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        semaphoreInfo.pNext = &exportInfo;

        if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &m_semaphore) != VK_SUCCESS) {
            std::cerr << "Failed to create Vulkan semaphore!" << std::endl;
            return;
        }

        int fd;
        VkSemaphoreGetFdInfoKHR semaphoreGetFdInfo = {};
        semaphoreGetFdInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
        semaphoreGetFdInfo.semaphore = m_semaphore;
        semaphoreGetFdInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;

        auto vkGetSemaphoreFdKHR = (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(device, "vkGetSemaphoreFdKHR");
        if (!vkGetSemaphoreFdKHR) {
             std::cerr << "Failed to load vkGetSemaphoreFdKHR!" << std::endl;
             return;
        }

        if (vkGetSemaphoreFdKHR(device, &semaphoreGetFdInfo, &fd) != VK_SUCCESS) {
            std::cerr << "Failed to get semaphore FD!" << std::endl;
            return;
        }

        cudaExternalSemaphoreHandleDesc externalSemaphoreHandleDesc = {};
        externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
        externalSemaphoreHandleDesc.handle.fd = fd;
        externalSemaphoreHandleDesc.flags = 0;

        if (cudaImportExternalSemaphore(&m_cuda_semaphore, &externalSemaphoreHandleDesc) != cudaSuccess) {
            std::cerr << "Failed to import external semaphore!" << std::endl;
        }
    }

    ~cuda_vulkan_semaphore() {
        if (m_cuda_semaphore) {
            cudaDestroyExternalSemaphore(m_cuda_semaphore);
        }
        if (m_semaphore) {
            vkDestroySemaphore(m_device, m_semaphore, nullptr);
        }
    }

    VkSemaphore get_vulkan_semaphore() const { return m_semaphore; }
    
    void wait(cudaStream_t stream) {
        cudaExternalSemaphoreWaitParams waitParams = {};
        waitParams.flags = 0;
        waitParams.params.fence.value = 0;
        cudaWaitExternalSemaphoresAsync(&m_cuda_semaphore, &waitParams, 1, stream);
    }

    void signal(cudaStream_t stream) {
        cudaExternalSemaphoreSignalParams signalParams = {};
        signalParams.flags = 0;
        signalParams.params.fence.value = 0;
        cudaSignalExternalSemaphoresAsync(&m_cuda_semaphore, &signalParams, 1, stream);
    }

private:
    VkDevice m_device;
    VkSemaphore m_semaphore = VK_NULL_HANDLE;
    cudaExternalSemaphore_t m_cuda_semaphore = nullptr;
};