#include "renderer.h"
#include "logger.h"
#include "application.h"
#include <fstream>
#include <array>

namespace utils {
    static uint32_t find_memory_type(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(application::get_physical_device(), &memProperties);

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        throw std::runtime_error("failed to find suitable memory type!");
    }
}

renderer::renderer(uint32_t width, uint32_t height, uint32_t depth) 
    : m_width(width), m_height(height), m_sim_width(width), m_sim_height(height), m_sim_depth(depth) {
    m_device = application::get_device();
#ifdef _DEBUG
    LOG_INFO("[Renderer] Initializing renderer with width: {} height: {} depth: {}", width, height, depth);
#endif
    init_vulkan_resources();

    m_camera = std::make_shared<camera>(45.0f, 0.1f, 100.0f, glm::vec3(0.0f, 0.0f, 2.0f), glm::vec3(0.0f, 0.0f, -1.0f));
    m_camera->resize(width, height);
}

renderer::~renderer() {
    vkDeviceWaitIdle(m_device);

    vkDeviceWaitIdle(m_device);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        if (m_render_fences[i]) vkDestroyFence(m_device, m_render_fences[i], nullptr);
        if (m_render_finished_semaphores[i]) vkDestroySemaphore(m_device, m_render_finished_semaphores[i], nullptr);
    }
    if (m_framebuffer) vkDestroyFramebuffer(m_device, m_framebuffer, nullptr);
    if (m_background_pipeline) vkDestroyPipeline(m_device, m_background_pipeline, nullptr);
    if (m_streamlines_pipeline) vkDestroyPipeline(m_device, m_streamlines_pipeline, nullptr);
    if (m_pipeline_layout) vkDestroyPipelineLayout(m_device, m_pipeline_layout, nullptr);
    if (m_descriptor_pool) vkDestroyDescriptorPool(m_device, m_descriptor_pool, nullptr);
    if (m_descriptor_set_layout) vkDestroyDescriptorSetLayout(m_device, m_descriptor_set_layout, nullptr);
    if (m_render_pass) vkDestroyRenderPass(m_device, m_render_pass, nullptr);
    if (m_command_pool) vkDestroyCommandPool(m_device, m_command_pool, nullptr);

    if (m_velocity_buffer) vkDestroyBuffer(m_device, m_velocity_buffer, nullptr);
    if (m_velocity_memory) vkFreeMemory(m_device, m_velocity_memory, nullptr);

    if (m_input_sampler) vkDestroySampler(m_device, m_input_sampler, nullptr);

    if (m_velocity_image_view) vkDestroyImageView(m_device, m_velocity_image_view, nullptr);
    if (m_velocity_image) vkDestroyImage(m_device, m_velocity_image, nullptr);
    if (m_velocity_image_memory) vkFreeMemory(m_device, m_velocity_image_memory, nullptr);

    if (m_solid_buffer) vkDestroyBuffer(m_device, m_solid_buffer, nullptr);
    if (m_solid_memory) vkFreeMemory(m_device, m_solid_memory, nullptr);

    if (m_solid_image_view) vkDestroyImageView(m_device, m_solid_image_view, nullptr);
    if (m_solid_image) vkDestroyImage(m_device, m_solid_image, nullptr);
    if (m_solid_image_memory) vkFreeMemory(m_device, m_solid_image_memory, nullptr);

    if (m_camera_buffer) vkDestroyBuffer(m_device, m_camera_buffer, nullptr);
    if (m_camera_memory) vkFreeMemory(m_device, m_camera_memory, nullptr);
}

void renderer::init_vulkan_resources() {
    // 1. Create Velocity Interop Buffer
    // Size = width * height * depth * sizeof(float4) (Using float4 for alignment/simplicity in 3D)
    // Size = width * height * depth * sizeof(float4) (Using float4 for alignment/simplicity in 3D)
    size_t velocity_buffer_size = m_sim_width * m_sim_height * m_sim_depth * sizeof(float) * 4;
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = velocity_buffer_size;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkExternalMemoryBufferCreateInfo externalMemoryBufferInfo = {};
    externalMemoryBufferInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    externalMemoryBufferInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    bufferInfo.pNext = &externalMemoryBufferInfo;

    check_vk_result(vkCreateBuffer(m_device, &bufferInfo, nullptr, &m_velocity_buffer));

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(m_device, m_velocity_buffer, &memRequirements);

    VkExportMemoryAllocateInfo exportAllocInfo = {};
    exportAllocInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    exportAllocInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = utils::find_memory_type(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    allocInfo.pNext = &exportAllocInfo;

    check_vk_result(vkAllocateMemory(m_device, &allocInfo, nullptr, &m_velocity_memory));
    check_vk_result(vkBindBufferMemory(m_device, m_velocity_buffer, m_velocity_memory, 0));

    // 2. Create Input Image (R32F) - REMOVED
    {
        // Create Sampler
        VkSamplerCreateInfo samplerInfo = {};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.anisotropyEnable = VK_FALSE;
        samplerInfo.maxAnisotropy = 1.0f;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

        check_vk_result(vkCreateSampler(m_device, &samplerInfo, nullptr, &m_input_sampler));
    }

    // 2b. Create Velocity Image (R32G32F)
    {
        VkImageCreateInfo imageInfo = {};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_3D;
        imageInfo.extent.width = m_sim_width;
        imageInfo.extent.height = m_sim_height;
        imageInfo.extent.depth = m_sim_depth;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = VK_FORMAT_R32G32B32A32_SFLOAT; // Use float4 for velocity in 3D
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        check_vk_result(vkCreateImage(m_device, &imageInfo, nullptr, &m_velocity_image));

        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(m_device, m_velocity_image, &memRequirements);

        VkMemoryAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = utils::find_memory_type(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        check_vk_result(vkAllocateMemory(m_device, &allocInfo, nullptr, &m_velocity_image_memory));
        check_vk_result(vkBindImageMemory(m_device, m_velocity_image, m_velocity_image_memory, 0));

        // Create Image View
        VkImageViewCreateInfo viewInfo = {};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = m_velocity_image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_3D;
        viewInfo.format = VK_FORMAT_R32G32B32A32_SFLOAT;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        check_vk_result(vkCreateImageView(m_device, &viewInfo, nullptr, &m_velocity_image_view));
    }

    // 2c. Create Solid Interop Buffer & Image (R8_UNORM)
    {
        // Buffer
        size_t solid_buffer_size = m_sim_width * m_sim_height * m_sim_depth * sizeof(uint8_t);
        VkBufferCreateInfo bufferInfo = {};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = solid_buffer_size;
        bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkExternalMemoryBufferCreateInfo externalMemoryBufferInfo = {};
        externalMemoryBufferInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
        externalMemoryBufferInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        bufferInfo.pNext = &externalMemoryBufferInfo;

        check_vk_result(vkCreateBuffer(m_device, &bufferInfo, nullptr, &m_solid_buffer));

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(m_device, m_solid_buffer, &memRequirements);

        VkExportMemoryAllocateInfo exportAllocInfo = {};
        exportAllocInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
        exportAllocInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

        VkMemoryAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = utils::find_memory_type(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        allocInfo.pNext = &exportAllocInfo;

        check_vk_result(vkAllocateMemory(m_device, &allocInfo, nullptr, &m_solid_memory));
        check_vk_result(vkBindBufferMemory(m_device, m_solid_buffer, m_solid_memory, 0));

        // Image
        VkImageCreateInfo imageInfo = {};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_3D;
        imageInfo.extent.width = m_sim_width;
        imageInfo.extent.height = m_sim_height;
        imageInfo.extent.depth = m_sim_depth;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = VK_FORMAT_R8_UNORM;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        check_vk_result(vkCreateImage(m_device, &imageInfo, nullptr, &m_solid_image));

        vkGetImageMemoryRequirements(m_device, m_solid_image, &memRequirements);
        
        // Remove export info for image memory (not exported)
        allocInfo.pNext = nullptr; 
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = utils::find_memory_type(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        check_vk_result(vkAllocateMemory(m_device, &allocInfo, nullptr, &m_solid_image_memory));
        check_vk_result(vkBindImageMemory(m_device, m_solid_image, m_solid_image_memory, 0));

        // Image View
        VkImageViewCreateInfo viewInfo = {};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = m_solid_image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_3D;
        viewInfo.format = VK_FORMAT_R8_UNORM;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        check_vk_result(vkCreateImageView(m_device, &viewInfo, nullptr, &m_solid_image_view));
    }

    // 2d. Create Camera Uniform Buffer (Dynamic)
    {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(application::get_physical_device(), &properties);
        size_t minUboAlignment = properties.limits.minUniformBufferOffsetAlignment;
        
        m_camera_buffer_size = sizeof(device_camera);
        if (minUboAlignment > 0) {
            m_camera_buffer_size = (m_camera_buffer_size + minUboAlignment - 1) & ~(minUboAlignment - 1);
        }

        VkBufferCreateInfo bufferInfo = {};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = m_camera_buffer_size * MAX_FRAMES_IN_FLIGHT;
        bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        check_vk_result(vkCreateBuffer(m_device, &bufferInfo, nullptr, &m_camera_buffer));

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(m_device, m_camera_buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = utils::find_memory_type(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        check_vk_result(vkAllocateMemory(m_device, &allocInfo, nullptr, &m_camera_memory));
        check_vk_result(vkBindBufferMemory(m_device, m_camera_buffer, m_camera_memory, 0));

        check_vk_result(vkMapMemory(m_device, m_camera_memory, 0, memRequirements.size, 0, &m_camera_mapped_memory));
    }

    // 3. Create Output Image
    m_output_image = std::make_shared<image>(m_width, m_height, image_type::rgba);

    // 4. Create Pipeline & RenderPass
    create_render_pass();
    create_pipeline();
    create_framebuffers();
    create_command_buffers();
}

void renderer::create_render_pass() {
    VkAttachmentDescription colorAttachment = {};
    colorAttachment.format = VK_FORMAT_R32G32B32A32_SFLOAT; // Output format
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; // Ready for ImGui

    VkAttachmentReference colorAttachmentRef = {};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;

    VkSubpassDependency dependency = {};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo renderPassInfo = {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &colorAttachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    check_vk_result(vkCreateRenderPass(m_device, &renderPassInfo, nullptr, &m_render_pass));
}

void renderer::create_pipeline() {
    // Descriptor Set Layout
    VkDescriptorSetLayoutBinding bindings[3] = {};
    
    // Binding 0: Velocity
    bindings[0].binding = 0;
    bindings[0].descriptorCount = 1;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[0].pImmutableSamplers = nullptr;
    bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // Binding 1: Solid
    bindings[1].binding = 1;
    bindings[1].descriptorCount = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[1].pImmutableSamplers = nullptr;
    bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // Binding 2: Camera
    bindings[2].binding = 2;
    bindings[2].descriptorCount = 1;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    bindings[2].pImmutableSamplers = nullptr;
    bindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 3;
    layoutInfo.pBindings = bindings;

    check_vk_result(vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_descriptor_set_layout));

    // Pipeline Layout
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &m_descriptor_set_layout;

    check_vk_result(vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr, &m_pipeline_layout));

    // --- Common Pipeline State ---
    auto vertShaderModule = create_shader_module_from_file("Engine/src/Graphics/Shaders/quad.vert.spv");
    
    VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vertShaderModule;
    vertShaderStageInfo.pName = "main";

    VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkPipelineViewportStateCreateInfo viewportState = {};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    std::vector<VkDynamicState> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };

    VkPipelineDynamicStateCreateInfo dynamicState = {};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    VkPipelineRasterizationStateCreateInfo rasterizer = {};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;

    VkPipelineMultisampleStateCreateInfo multisampling = {};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // --- 1. Raymarching Pipeline ---
    {
        auto fragShaderModule = create_shader_module_from_file("Engine/src/Graphics/Shaders/raymarch.frag.spv");

        VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

        VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE; // Raymarching shader handles blending internally/outputs final color

        VkPipelineColorBlendStateCreateInfo colorBlending = {};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;

        VkGraphicsPipelineCreateInfo pipelineInfo = {};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.layout = m_pipeline_layout;
        pipelineInfo.renderPass = m_render_pass;
        pipelineInfo.subpass = 0;

        check_vk_result(vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_background_pipeline)); // Reusing m_background_pipeline handle for now
        vkDestroyShaderModule(m_device, fragShaderModule, nullptr);
    }

    vkDestroyShaderModule(m_device, vertShaderModule, nullptr);

    // Descriptor Pool & Set
    VkDescriptorPoolSize poolSize = {};
    poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSize.descriptorCount = 2 * MAX_FRAMES_IN_FLIGHT;

    VkDescriptorPoolSize poolSizeUniform = {};
    poolSizeUniform.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    poolSizeUniform.descriptorCount = 1 * MAX_FRAMES_IN_FLIGHT;

    VkDescriptorPoolSize poolSizes[] = {poolSize, poolSizeUniform};

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 2;
    poolInfo.pPoolSizes = poolSizes;
    poolInfo.maxSets = MAX_FRAMES_IN_FLIGHT;

    check_vk_result(vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descriptor_pool));

    std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, m_descriptor_set_layout);
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = m_descriptor_pool;
    allocInfo.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
    allocInfo.pSetLayouts = layouts.data();

    m_descriptor_sets.resize(MAX_FRAMES_IN_FLIGHT);
    check_vk_result(vkAllocateDescriptorSets(m_device, &allocInfo, m_descriptor_sets.data()));

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        VkDescriptorImageInfo velocityInfo = {};
        velocityInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        velocityInfo.imageView = m_velocity_image_view;
        velocityInfo.sampler = m_input_sampler;

        VkDescriptorImageInfo solidInfo = {};
        solidInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        solidInfo.imageView = m_solid_image_view;
        solidInfo.sampler = m_input_sampler;

        VkDescriptorBufferInfo cameraInfo = {};
        cameraInfo.buffer = m_camera_buffer;
        cameraInfo.offset = 0;
        cameraInfo.range = sizeof(device_camera);

        VkWriteDescriptorSet descriptorWrites[3] = {};

        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = m_descriptor_sets[i];
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pImageInfo = &velocityInfo;

        descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = m_descriptor_sets[i];
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].dstArrayElement = 0;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pImageInfo = &solidInfo;

        descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[2].dstSet = m_descriptor_sets[i];
        descriptorWrites[2].dstBinding = 2;
        descriptorWrites[2].dstArrayElement = 0;
        descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        descriptorWrites[2].descriptorCount = 1;
        descriptorWrites[2].pBufferInfo = &cameraInfo;

        vkUpdateDescriptorSets(m_device, 3, descriptorWrites, 0, nullptr);
    }
}

void renderer::create_framebuffers() {
    VkImageView outputView = m_output_image->get_view();
    
    VkFramebufferCreateInfo framebufferInfo = {};
    framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferInfo.renderPass = m_render_pass;
    framebufferInfo.attachmentCount = 1;
    framebufferInfo.pAttachments = &outputView;
    framebufferInfo.width = m_width;
    framebufferInfo.height = m_height;
    framebufferInfo.layers = 1;

    check_vk_result(vkCreateFramebuffer(m_device, &framebufferInfo, nullptr, &m_framebuffer));
}

void renderer::create_command_buffers() {
    uint32_t queueFamilyIndex = application::get_graphics_queue_family_index();

    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = queueFamilyIndex;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    check_vk_result(vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_command_pool));

    m_command_buffers.resize(MAX_FRAMES_IN_FLIGHT);

    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = m_command_pool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = (uint32_t)m_command_buffers.size();

    check_vk_result(vkAllocateCommandBuffers(m_device, &allocInfo, m_command_buffers.data()));

    // Transfer command buffer
    allocInfo.commandBufferCount = 1;
    check_vk_result(vkAllocateCommandBuffers(m_device, &allocInfo, &m_transfer_command_buffer));

    // Synchronization objects
    m_render_fences.resize(MAX_FRAMES_IN_FLIGHT);
    m_render_finished_semaphores.resize(MAX_FRAMES_IN_FLIGHT);

    VkSemaphoreCreateInfo semaphoreInfo = {};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        check_vk_result(vkCreateSemaphore(m_device, &semaphoreInfo, nullptr, &m_render_finished_semaphores[i]));
        check_vk_result(vkCreateFence(m_device, &fenceInfo, nullptr, &m_render_fences[i]));
    }
}

VkShaderModule renderer::create_shader_module(const std::vector<uint32_t>& code) {
    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size() * sizeof(uint32_t);
    createInfo.pCode = code.data();

    VkShaderModule shaderModule;
    check_vk_result(vkCreateShaderModule(m_device, &createInfo, nullptr, &shaderModule));

    return shaderModule;
}

VkShaderModule renderer::create_shader_module_from_file(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file: " + filepath);
    }

    size_t fileSize = (size_t)file.tellg();
    std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));

    file.seekg(0);
    file.read((char*)buffer.data(), fileSize);
    file.close();

    return create_shader_module(buffer);
}

// get_density_fd removed

int renderer::get_velocity_fd() {
    int fd;
    VkMemoryGetFdInfoKHR memoryGetFdInfo = {};
    memoryGetFdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    memoryGetFdInfo.memory = m_velocity_memory;
    memoryGetFdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    auto vkGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(m_device, "vkGetMemoryFdKHR");
    if (!vkGetMemoryFdKHR) {
        throw std::runtime_error("Failed to load vkGetMemoryFdKHR");
    }

    check_vk_result(vkGetMemoryFdKHR(m_device, &memoryGetFdInfo, &fd));
    return fd;
}

int renderer::get_solid_fd() {
    int fd;
    VkMemoryGetFdInfoKHR memoryGetFdInfo = {};
    memoryGetFdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    memoryGetFdInfo.memory = m_solid_memory;
    memoryGetFdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    auto vkGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(m_device, "vkGetMemoryFdKHR");
    if (!vkGetMemoryFdKHR) {
        throw std::runtime_error("Failed to load vkGetMemoryFdKHR");
    }

    check_vk_result(vkGetMemoryFdKHR(m_device, &memoryGetFdInfo, &fd));
    return fd;
}

void renderer::update_sim_data() {
    cudaDeviceSynchronize();
    vkResetCommandBuffer(m_transfer_command_buffer, 0);

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    check_vk_result(vkBeginCommandBuffer(m_transfer_command_buffer, &beginInfo));

    // Transition Images to Transfer Dst
    // Transition Images to Transfer Dst
    VkImageMemoryBarrier barriers[3] = {};
    
    // Density (Not used in this pass but kept for consistency if needed, or we can skip)
    // We skip density for now as we are focusing on Velocity + Solid

    // Velocity
    barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barriers[0].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barriers[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[0].image = m_velocity_image;
    barriers[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barriers[0].subresourceRange.baseMipLevel = 0;
    barriers[0].subresourceRange.levelCount = 1;
    barriers[0].subresourceRange.baseArrayLayer = 0;
    barriers[0].subresourceRange.layerCount = 1;
    barriers[0].srcAccessMask = 0;
    barriers[0].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    // Solid
    barriers[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barriers[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barriers[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barriers[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[1].image = m_solid_image;
    barriers[1].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barriers[1].subresourceRange.baseMipLevel = 0;
    barriers[1].subresourceRange.levelCount = 1;
    barriers[1].subresourceRange.baseArrayLayer = 0;
    barriers[1].subresourceRange.layerCount = 1;
    barriers[1].srcAccessMask = 0;
    barriers[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    vkCmdPipelineBarrier(m_transfer_command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 2, barriers);

    // Copy Velocity
    VkBufferImageCopy region = {};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {(uint32_t)m_sim_width, (uint32_t)m_sim_height, (uint32_t)m_sim_depth};

    vkCmdCopyBufferToImage(m_transfer_command_buffer, m_velocity_buffer, m_velocity_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // Copy Solid
    vkCmdCopyBufferToImage(m_transfer_command_buffer, m_solid_buffer, m_solid_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // Transition to Shader Read
    barriers[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barriers[0].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barriers[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    barriers[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barriers[1].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barriers[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barriers[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(m_transfer_command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 2, barriers);

    check_vk_result(vkEndCommandBuffer(m_transfer_command_buffer));

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &m_transfer_command_buffer;

    VkQueue graphicsQueue = application::get_graphics_queue();

    check_vk_result(vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE));
    check_vk_result(vkQueueWaitIdle(graphicsQueue));
}

void renderer::update_camera(float dt) {
    if (m_camera->on_update(dt)) {
        // Camera updated
    }
    
    // Always update the buffer for the current frame
    device_camera cam_data;
    m_camera->allocate_device_resources(cam_data);

    char* mappedData = (char*)m_camera_mapped_memory;
    memcpy(mappedData + (m_current_frame * m_camera_buffer_size), &cam_data, sizeof(device_camera));
}

void renderer::render() {
    vkWaitForFences(m_device, 1, &m_render_fences[m_current_frame], VK_TRUE, UINT64_MAX);
    vkResetFences(m_device, 1, &m_render_fences[m_current_frame]);

    VkCommandBuffer commandBuffer = m_command_buffers[m_current_frame];

    vkResetCommandBuffer(commandBuffer, 0);

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    check_vk_result(vkBeginCommandBuffer(commandBuffer, &beginInfo));

    VkRenderPassBeginInfo renderPassInfo = {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = m_render_pass;
    renderPassInfo.framebuffer = m_framebuffer;
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = {(uint32_t)m_width, (uint32_t)m_height};

    VkClearValue clearColor = {0.0f, 0.0f, 0.0f, 1.0f};
    renderPassInfo.clearValueCount = 1;
    renderPassInfo.pClearValues = &clearColor;

    vkCmdBeginRenderPass(m_command_buffers[m_current_frame], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(m_command_buffers[m_current_frame], VK_PIPELINE_BIND_POINT_GRAPHICS, m_background_pipeline); // Using background pipeline handle for raymarching

    VkViewport viewport = {};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float)m_width;
    viewport.height = (float)m_height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(m_command_buffers[m_current_frame], 0, 1, &viewport);

    VkRect2D scissor = {};
    scissor.offset = {0, 0};
    scissor.extent = {(uint32_t)m_width, (uint32_t)m_height};
    vkCmdSetScissor(m_command_buffers[m_current_frame], 0, 1, &scissor);

    uint32_t dynamicOffset = m_current_frame * m_camera_buffer_size;
    vkCmdBindDescriptorSets(m_command_buffers[m_current_frame], VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline_layout, 0, 1, &m_descriptor_sets[m_current_frame], 1, &dynamicOffset);

    vkCmdDraw(m_command_buffers[m_current_frame], 6, 1, 0, 0);

    vkCmdEndRenderPass(m_command_buffers[m_current_frame]);

    check_vk_result(vkEndCommandBuffer(commandBuffer));

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    VkSemaphore signalSemaphores[] = {m_render_finished_semaphores[m_current_frame]};
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    VkQueue graphicsQueue = application::get_graphics_queue();

    check_vk_result(vkQueueSubmit(graphicsQueue, 1, &submitInfo, m_render_fences[m_current_frame]));

    // Synchronize with Main App
    application::get().add_wait_semaphore(m_render_finished_semaphores[m_current_frame]);

    m_current_frame = (m_current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
}

void renderer::resize(uint32_t width, uint32_t height) {
    if (m_output_image && m_output_image->get_width() == width &&
        m_output_image->get_height() == height) {
        return;
    }

#ifdef _DEBUG
    try {
        LOG_INFO("Resizing renderer to {}x{}", width, height);
        m_output_image = std::make_shared<image>(width, height, image_type::rgba);
    }
    catch (const std::exception& e) {
        LOG_ERROR("Failed to create output image: {}", e.what());
        return;
    }

    if (!m_output_image) {
        LOG_ERROR("Failed to create output image");
        return;
    }
#else
    m_output_image = std::make_shared<image>(width, height, image_type::rgba);
#endif

    m_width = width;
    m_height = height;

    if (m_camera) {
        m_camera->resize(width, height);
    }

    if (m_framebuffer) {
        vkDestroyFramebuffer(m_device, m_framebuffer, nullptr);
    }
    create_framebuffers();
}
