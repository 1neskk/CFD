#include "image.h"

#include "application.h"
#include "backends/imgui_impl_vulkan.h"
#include "imgui.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace utils {
static uint32_t get_vulkan_memory_type(VkMemoryPropertyFlags properties,
                                       uint32_t typeBits) {
    VkPhysicalDeviceMemoryProperties memoryProperties;
    vkGetPhysicalDeviceMemoryProperties(application::get_physical_device(),
                                        &memoryProperties);

    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++) {
        if ((memoryProperties.memoryTypes[i].propertyFlags & properties) ==
                properties &&
            typeBits & (1 << i))
            return i;
    }
    return 0xffffffff;
}

static uint32_t bytes_per_pixel(image_type format) {
    switch (format) {
        case image_type::rgba:
            return 4;
        case image_type::rgba32f:
            return 16;
        case image_type::none:
            return 0;
    }
    return 0;
}

static VkFormat get_vulkan_format(image_type format) {
    switch (format) {
        case image_type::rgba:
            return VK_FORMAT_R8G8B8A8_UNORM;
        case image_type::rgba32f:
            return VK_FORMAT_R32G32B32A32_SFLOAT;
        case image_type::none:
            return static_cast<VkFormat>(0);
    }
    return static_cast<VkFormat>(0);
}
}  // namespace utils

image::image(std::string_view path) : m_filepath(path) {
    int width, height, channels;
    uint8_t *data = nullptr;

    if (stbi_is_hdr(m_filepath.c_str())) {
        data = reinterpret_cast<uint8_t *>(stbi_loadf(
            m_filepath.c_str(), &width, &height, &channels, STBI_rgb_alpha));
        m_type = image_type::rgba32f;
    } else {
        data = stbi_load(m_filepath.c_str(), &width, &height, &channels,
                         STBI_rgb_alpha);
        m_type = image_type::rgba;
    }

    m_width = width;
    m_height = height;

    allocate_memory(m_width * m_height * utils::bytes_per_pixel(m_type));
    set_data(data);
    stbi_image_free(data);
}

image::image(uint32_t width, uint32_t height, image_type type, const void *data)
    : m_width(width), m_height(height), m_type(type) {
    allocate_memory(m_width * m_height * utils::bytes_per_pixel(m_type));
    if (data) set_data(data);
}

image::~image() { release(); }

void image::allocate_memory(uint64_t /*size*/) {
    VkDevice device = application::get_device();
    VkResult err;

    VkFormat format = utils::get_vulkan_format(m_type);

    // Image
    {
        VkImageCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        info.imageType = VK_IMAGE_TYPE_2D;
        info.format = format;
        info.extent.width = m_width;
        info.extent.height = m_height;
        info.extent.depth = 1;
        info.mipLevels = 1;
        info.arrayLayers = 1;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.usage = VK_IMAGE_USAGE_SAMPLED_BIT |
                     VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                     VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        err = vkCreateImage(device, &info, nullptr, &m_image);
        check_vk_result(err);
        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(device, m_image, &req);
        VkMemoryAllocateInfo alloc_info = {};
        alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc_info.allocationSize = req.size;
        alloc_info.memoryTypeIndex = utils::get_vulkan_memory_type(
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, req.memoryTypeBits);
        err = vkAllocateMemory(device, &alloc_info, nullptr, &m_memory);
        check_vk_result(err);
        err = vkBindImageMemory(device, m_image, m_memory, 0);
        check_vk_result(err);
    }

    // Image View
    {
        VkImageViewCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        info.image = m_image;
        info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        info.format = format;
        info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        info.subresourceRange.levelCount = 1;
        info.subresourceRange.layerCount = 1;
        err = vkCreateImageView(device, &info, nullptr, &m_image_view);
        check_vk_result(err);
    }

    // Sampler
    {
        VkSamplerCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        info.magFilter = VK_FILTER_LINEAR;
        info.minFilter = VK_FILTER_LINEAR;
        info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        info.minLod = -1000;
        info.maxLod = 1000;
        info.maxAnisotropy = 1.0f;
        VkResult err = vkCreateSampler(device, &info, nullptr, &m_sampler);
        check_vk_result(err);
    }

    m_descriptor_set =
        reinterpret_cast<VkDescriptorSet>(ImGui_ImplVulkan_AddTexture(
            m_sampler, m_image_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
}

void image::release() {
    application::submit_resource_free(
        [sampler = m_sampler, imageView = m_image_view, image = m_image,
         memory = m_memory, stagingBuffer = m_buffer,
         stagingBufferMemory = m_buffer_memory]() {
            VkDevice device = application::get_device();

            vkDestroySampler(device, sampler, nullptr);
            vkDestroyImageView(device, imageView, nullptr);
            vkDestroyImage(device, image, nullptr);
            vkFreeMemory(device, memory, nullptr);
            vkDestroyBuffer(device, stagingBuffer, nullptr);
            vkFreeMemory(device, stagingBufferMemory, nullptr);
        });

    m_sampler = nullptr;
    m_image_view = nullptr;
    m_image = nullptr;
    m_memory = nullptr;
    m_buffer = nullptr;
    m_buffer_memory = nullptr;
}

void image::set_data(const void *data) {
    VkDevice device = application::get_device();
    VkResult err;

    size_t size = m_width * m_height * utils::bytes_per_pixel(m_type);

    // Staging Buffer
    if (!m_buffer) {
        {
            VkBufferCreateInfo buffer_info = {};
            buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            buffer_info.size = size;
            buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            err = vkCreateBuffer(device, &buffer_info, nullptr, &m_buffer);
            check_vk_result(err);
            VkMemoryRequirements req;
            vkGetBufferMemoryRequirements(device, m_buffer, &req);
            m_size = req.size;
            VkMemoryAllocateInfo alloc_info = {};
            alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            alloc_info.allocationSize = req.size;
            alloc_info.memoryTypeIndex = utils::get_vulkan_memory_type(
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, req.memoryTypeBits);
            err = vkAllocateMemory(device, &alloc_info, nullptr,
                                   &m_buffer_memory);
            check_vk_result(err);
            err = vkBindBufferMemory(device, m_buffer, m_buffer_memory, 0);
            check_vk_result(err);
        }
    }

    // Copy data to staging buffer
    {
        char *map = nullptr;
        err = vkMapMemory(device, m_buffer_memory, 0, m_size, 0,
                          reinterpret_cast<void **>(&map));
        check_vk_result(err);
        memcpy(map, data, size);
        VkMappedMemoryRange range[1] = {};
        range[0].sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range[0].memory = m_buffer_memory;
        range[0].size = m_size;
        err = vkFlushMappedMemoryRanges(device, 1, range);
        check_vk_result(err);
        vkUnmapMemory(device, m_buffer_memory);
    }

    // Copy to image
    {
        VkCommandBuffer command_buffer = application::get_command_buffer();

        VkImageMemoryBarrier copy_barrier = {};
        copy_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        copy_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        copy_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        copy_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        copy_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        copy_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        copy_barrier.image = m_image;
        copy_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy_barrier.subresourceRange.levelCount = 1;
        copy_barrier.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_HOST_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                             nullptr, 1, &copy_barrier);

        VkBufferImageCopy region = {};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent.width = m_width;
        region.imageExtent.height = m_height;
        region.imageExtent.depth = 1;
        vkCmdCopyBufferToImage(command_buffer, m_buffer, m_image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                               &region);

        VkImageMemoryBarrier use_barrier = {};
        use_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        use_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        use_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        use_barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        use_barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        use_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        use_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        use_barrier.image = m_image;
        use_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        use_barrier.subresourceRange.levelCount = 1;
        use_barrier.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                             nullptr, 0, nullptr, 1, &use_barrier);

        application::flush_command_buffer(command_buffer);
    }
}

void image::resize(uint32_t width, uint32_t height) {
    if (m_image && m_width == width && m_height == height) return;

    m_width = width;
    m_height = height;

    release();
    allocate_memory(m_width * m_height * utils::bytes_per_pixel(m_type));
}
