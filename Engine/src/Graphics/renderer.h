#pragma once

#include <memory>
#include <cstdint>

#include "image.h"
#include "vulkan_cuda_buffer.cuh"
#include "cuda_buffer.cuh"
#include "vk_cuda_semaphore.h"

class renderer {
public:
    renderer(uint32_t width, uint32_t height);
    ~renderer();

    void update_sim_data();
    void render();
    
    std::shared_ptr<image> get_output_image() const { return m_output_image; }
    void resize(uint32_t width, uint32_t height);
    int get_density_fd();
    int get_velocity_fd();
    int get_solid_fd();

private:
    void init_vulkan_resources();
    void create_render_pass();
    void create_pipeline();
    void create_framebuffers();
    void create_command_buffers();

    VkShaderModule create_shader_module(const std::vector<uint32_t>& code);
    VkShaderModule create_shader_module_from_file(const std::string& filepath);

private:
    uint32_t m_width;
    uint32_t m_height;
    uint32_t m_sim_width;
    uint32_t m_sim_height;
    VkDevice m_device;

    // Interop
    VkBuffer m_density_buffer = VK_NULL_HANDLE;
    VkDeviceMemory m_density_memory = VK_NULL_HANDLE;
    VkBuffer m_velocity_buffer = VK_NULL_HANDLE;
    VkDeviceMemory m_velocity_memory = VK_NULL_HANDLE;
    VkBuffer m_solid_buffer = VK_NULL_HANDLE;
    VkDeviceMemory m_solid_memory = VK_NULL_HANDLE;

    // Resources
    VkImage m_input_image = VK_NULL_HANDLE;
    VkDeviceMemory m_input_image_memory = VK_NULL_HANDLE;
    VkImageView m_input_image_view = VK_NULL_HANDLE;
    VkSampler m_input_sampler = VK_NULL_HANDLE;

    VkImage m_velocity_image = VK_NULL_HANDLE;
    VkDeviceMemory m_velocity_image_memory = VK_NULL_HANDLE;
    VkImageView m_velocity_image_view = VK_NULL_HANDLE;

    VkImage m_solid_image = VK_NULL_HANDLE;
    VkDeviceMemory m_solid_image_memory = VK_NULL_HANDLE;
    VkImageView m_solid_image_view = VK_NULL_HANDLE;

    std::shared_ptr<image> m_output_image;

    // Pipeline
    VkRenderPass m_render_pass = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_descriptor_set_layout = VK_NULL_HANDLE;
    VkPipelineLayout m_pipeline_layout = VK_NULL_HANDLE;
    VkPipeline m_background_pipeline = VK_NULL_HANDLE;
    VkPipeline m_streamlines_pipeline = VK_NULL_HANDLE;
    VkDescriptorPool m_descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet m_descriptor_set = VK_NULL_HANDLE;

    VkFramebuffer m_framebuffer = VK_NULL_HANDLE;
    VkCommandPool m_command_pool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> m_command_buffers;
    VkCommandBuffer m_transfer_command_buffer = VK_NULL_HANDLE;
    std::vector<VkFence> m_render_fences;
    std::vector<VkSemaphore> m_render_finished_semaphores;
    
    uint32_t m_current_frame = 0;
    static constexpr int MAX_FRAMES_IN_FLIGHT = 2;
};