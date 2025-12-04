#pragma once

#include <string>

#include "vulkan/vulkan.h"

enum class image_type { none = 0, rgba, rgba32f };

class image {
   public:
    image(std::string_view path);
    image(uint32_t width, uint32_t height, image_type type,
          const void *data = nullptr);
    ~image();

    void set_data(const void *data);

    VkDescriptorSet get_descriptor_set() const { return m_descriptor_set; }

    void resize(uint32_t width, uint32_t height);

    uint32_t get_width() const { return m_width; }
    uint32_t get_height() const { return m_height; }

    VkImage get_handle() const { return m_image; }
    VkImageView get_view() const { return m_image_view; }

   private:
    void allocate_memory(uint64_t size);
    void release();

   private:
    uint32_t m_width = 0, m_height = 0;

    VkDescriptorSet m_descriptor_set = nullptr;

    VkImage m_image = nullptr;
    VkImageView m_image_view = nullptr;
    VkDeviceMemory m_memory = nullptr;
    VkSampler m_sampler = nullptr;

    image_type m_type = image_type::none;

    VkBuffer m_buffer = nullptr;
    VkDeviceMemory m_buffer_memory = nullptr;

    size_t m_size = 0;

    std::string m_filepath;
};