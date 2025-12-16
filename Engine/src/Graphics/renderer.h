#pragma once

#include <memory>
#include <cstdint>

#include "image.h"
#include "vulkan_cuda_buffer.cuh"
#include "cuda_buffer.cuh"
#include "vk_cuda_semaphore.h"
#include "camera.h"

struct VectorPushConstants {
    uint32_t gridX;
    uint32_t gridY;
    uint32_t gridZ;
    float scale;
};

struct StreamlinePushConstants {
    uint32_t numSeeds;
    uint32_t numSteps;
    float stepSize;
    float seedY;
};

class renderer {
public:
    renderer(uint32_t width, uint32_t height, uint32_t depth);
    ~renderer();

    void update_sim_data();
    void update_camera(float dt);
    void render();
    void compute_streamlines();
    
    std::shared_ptr<image> get_output_image() const { return m_output_image; }
    void resize(uint32_t width, uint32_t height);
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
    uint32_t m_sim_depth;
    VkDevice m_device;

    // Interop
    VkBuffer m_velocity_buffer = VK_NULL_HANDLE;
    VkDeviceMemory m_velocity_memory = VK_NULL_HANDLE;
    VkBuffer m_solid_buffer = VK_NULL_HANDLE;
    VkDeviceMemory m_solid_memory = VK_NULL_HANDLE;

    // Resources
    VkSampler m_input_sampler = VK_NULL_HANDLE;

    VkImage m_velocity_image = VK_NULL_HANDLE;
    VkDeviceMemory m_velocity_image_memory = VK_NULL_HANDLE;
    VkImageView m_velocity_image_view = VK_NULL_HANDLE;

    VkImage m_solid_image = VK_NULL_HANDLE;
    VkDeviceMemory m_solid_image_memory = VK_NULL_HANDLE;
    VkImageView m_solid_image_view = VK_NULL_HANDLE;

    // Camera
    std::shared_ptr<camera> m_camera;
    VkBuffer m_camera_buffer = VK_NULL_HANDLE;
    VkDeviceMemory m_camera_memory = VK_NULL_HANDLE;
    void* m_camera_mapped_memory = nullptr;
    size_t m_camera_buffer_size = 0;

    // Vector Resources
    VkBuffer m_vector_vertex_buffer = VK_NULL_HANDLE;
    VkDeviceMemory m_vector_vertex_memory = VK_NULL_HANDLE;
    uint32_t m_vector_vertex_count = 0;

    // Streamline Resources
    VkBuffer m_streamline_vertex_buffer = VK_NULL_HANDLE;
    VkDeviceMemory m_streamline_vertex_memory = VK_NULL_HANDLE;
    uint32_t m_streamline_vertex_count = 0;
    
    std::shared_ptr<image> m_output_image;
    
    VkImage m_depth_image = VK_NULL_HANDLE;
    VkDeviceMemory m_depth_image_memory = VK_NULL_HANDLE;
    VkImageView m_depth_image_view = VK_NULL_HANDLE;

    // Pipeline
    VkRenderPass m_render_pass = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_descriptor_set_layout = VK_NULL_HANDLE;
    VkPipelineLayout m_pipeline_layout = VK_NULL_HANDLE;
    VkPipeline m_background_pipeline = VK_NULL_HANDLE;
    VkPipeline m_vector_pipeline = VK_NULL_HANDLE;
    VkPipeline m_streamline_graphics_pipeline = VK_NULL_HANDLE;
    VkPipeline m_streamline_compute_pipeline = VK_NULL_HANDLE;
    VkPipelineLayout m_streamline_compute_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_streamline_compute_descriptor_layout = VK_NULL_HANDLE;
    VkDescriptorPool m_streamline_descriptor_pool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> m_streamline_descriptor_sets;
    VkDescriptorPool m_descriptor_pool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> m_descriptor_sets;

    VkFramebuffer m_framebuffer = VK_NULL_HANDLE;
    VkCommandPool m_command_pool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> m_command_buffers;
    VkCommandBuffer m_transfer_command_buffer = VK_NULL_HANDLE;
    std::vector<VkFence> m_render_fences;
    std::vector<VkSemaphore> m_render_finished_semaphores;
    
    uint32_t m_current_frame = 0;
    static constexpr int MAX_FRAMES_IN_FLIGHT = 2;
};