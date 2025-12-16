#include "application.h"

#include <stdio.h>
#include <stdlib.h>

#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_vulkan.h"
#include "logger.h"
#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>

#include <iostream>

#include "backends/imgui_impl_glfw.cpp"
#include "imgui/Roboto-Regular.embed"
#include "imgui_internal.h"

extern bool g_running;

#ifdef _DEBUG
#define IMGUI_VULKAN_DEBUG_REPORT
#endif

static VkAllocationCallbacks *g_allocator = nullptr;
static VkInstance g_Instance = VK_NULL_HANDLE;
static VkPhysicalDevice g_PhysicalDevice = VK_NULL_HANDLE;
static VkDevice g_Device = VK_NULL_HANDLE;
static uint32_t g_QueueFamily = (uint32_t)-1;
static VkQueue g_Queue = VK_NULL_HANDLE;
static VkDebugReportCallbackEXT g_DebugReport = VK_NULL_HANDLE;
static VkPipelineCache g_PipelineCache = VK_NULL_HANDLE;
static VkDescriptorPool g_DescriptorPool = VK_NULL_HANDLE;

static ImGui_ImplVulkanH_Window g_MainWindowData;
static int g_MinImageCount = 2;
static bool g_SwapChainRebuild = false;

static std::vector<std::vector<VkCommandBuffer>> s_AllocatedCommandBuffers;
static std::vector<std::vector<std::function<void()>>> s_ResourceFreeQueue;

static uint32_t s_CurrentFrameIndex = 0;

static application *s_instance = nullptr;

void check_vk_result(VkResult result) {
    if (result == 0) return;
    fprintf(stderr, "[vulkan] Error: VkResult = %d\n", result);
    if (result < 0) abort();
}

#ifdef IMGUI_VULKAN_DEBUG_REPORT
static VKAPI_ATTR VkBool32 VKAPI_CALL debug_report(
    VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT objectType,
    uint64_t object, size_t location, int32_t messageCode,
    const char *p_layer_prefix, const char *pMessage, void *pUserData) {
    (void)flags;
    (void)object;
    (void)location;
    (void)messageCode;
    (void)pUserData;
    (void)p_layer_prefix;  // Unused arguments
    fprintf(stderr,
            "[vulkan] Debug report from ObjectType: %i\nMessage: %s\n\n",
            objectType, pMessage);
    return VK_FALSE;
}
#endif  // IMGUI_VULKAN_DEBUG_REPORT

static bool check_validation_layer_support(const char *layerName) {
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const auto &layerProperties : availableLayers) {
        if (strcmp(layerName, layerProperties.layerName) == 0) {
            return true;
        }
    }

    return false;
}

static void setup_vulkan(const char **extensions, uint32_t extensions_count) {
    VkResult err;

    {
        VkInstanceCreateInfo create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        create_info.enabledExtensionCount = extensions_count;
        create_info.ppEnabledExtensionNames = extensions;
#ifdef IMGUI_VULKAN_DEBUG_REPORT
        const char *validationLayer = "VK_LAYER_KHRONOS_validation";
        bool validationLayerPresent =
            check_validation_layer_support(validationLayer);

        if (validationLayerPresent) {
            const char *layers[] = {validationLayer};
            create_info.enabledLayerCount = 1;
            create_info.ppEnabledLayerNames = layers;
        } else {
            fprintf(stderr,
                    "[vulkan] Validation layer %s not present, disabling "
                    "validation.\n",
                    validationLayer);
            create_info.enabledLayerCount = 0;
        }

        const char **extensions_ext = static_cast<const char **>(
            malloc(sizeof(const char *) * (extensions_count + 4)));
        memcpy(extensions_ext, extensions,
               extensions_count * sizeof(const char *));
        extensions_ext[extensions_count] = "VK_EXT_debug_report";
        extensions_ext[extensions_count + 1] =
            VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME;
        extensions_ext[extensions_count + 2] =
            VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME;
        extensions_ext[extensions_count + 3] =
            VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME;
        create_info.enabledExtensionCount = extensions_count + 4;
        create_info.ppEnabledExtensionNames = extensions_ext;

        err = vkCreateInstance(&create_info, g_allocator, &g_Instance);
        check_vk_result(err);
        free(extensions_ext);

        // Only create debug report callback if validation layer is present
        // (usually they go together) actually VK_EXT_debug_report is an
        // extension, not a layer. But usually we only want it if we are
        // debugging.

        auto vkCreateDebugReportCallbackEXT =
            (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(
                g_Instance, "vkCreateDebugReportCallbackEXT");
        // IM_ASSERT(vkCreateDebugReportCallbackEXT != NULL);

        if (vkCreateDebugReportCallbackEXT) {
            VkDebugReportCallbackCreateInfoEXT debug_report_ci = {};
            debug_report_ci.sType =
                VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
            debug_report_ci.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT |
                                    VK_DEBUG_REPORT_WARNING_BIT_EXT |
                                    VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT;
            debug_report_ci.pfnCallback = debug_report;
            debug_report_ci.pUserData = nullptr;
            err = vkCreateDebugReportCallbackEXT(g_Instance, &debug_report_ci,
                                                 g_allocator, &g_DebugReport);
            check_vk_result(err);
        }
#else
        const char **extensions_ext = static_cast<const char **>(
            malloc(sizeof(const char *) * (extensions_count + 3)));
        memcpy(extensions_ext, extensions,
               extensions_count * sizeof(const char *));
        extensions_ext[extensions_count] =
            VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME;
        extensions_ext[extensions_count + 1] =
            VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME;
        extensions_ext[extensions_count + 2] =
            VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME;
        create_info.enabledExtensionCount = extensions_count + 3;
        create_info.ppEnabledExtensionNames = extensions_ext;

        err = vkCreateInstance(&create_info, g_allocator, &g_Instance);
        check_vk_result(err);
        free(extensions_ext);
        IM_UNUSED(g_DebugReport);
#endif
    }

    {
        uint32_t gpu_count;
        err = vkEnumeratePhysicalDevices(g_Instance, &gpu_count, NULL);
        check_vk_result(err);
        IM_ASSERT(gpu_count > 0);

        VkPhysicalDevice *gpus =
            (VkPhysicalDevice *)malloc(sizeof(VkPhysicalDevice) * gpu_count);
        err = vkEnumeratePhysicalDevices(g_Instance, &gpu_count, gpus);
        check_vk_result(err);

        int use_gpu = 0;
        for (int i = 0; i < (int)gpu_count; i++) {
            VkPhysicalDeviceProperties properties;
            vkGetPhysicalDeviceProperties(gpus[i], &properties);
            if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                use_gpu = i;
                break;
            }
        }

        g_PhysicalDevice = gpus[use_gpu];
        free(gpus);
    }

    {
        uint32_t count;
        vkGetPhysicalDeviceQueueFamilyProperties(g_PhysicalDevice, &count,
                                                 NULL);
        VkQueueFamilyProperties *queues = (VkQueueFamilyProperties *)malloc(
            sizeof(VkQueueFamilyProperties) * count);
        vkGetPhysicalDeviceQueueFamilyProperties(g_PhysicalDevice, &count,
                                                 queues);
        for (uint32_t i = 0; i < count; i++)
            if (queues[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                g_QueueFamily = i;
                break;
            }
        free(queues);
        IM_ASSERT(g_QueueFamily != (uint32_t)-1);
    }

    {
        int device_extension_count = 5;
        const char *device_extensions[] = {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME,
            VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
            VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
            VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
            VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME};
        const float queue_priority[] = {1.0f};
        VkDeviceQueueCreateInfo queue_info[1] = {};
        queue_info[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_info[0].queueFamilyIndex = g_QueueFamily;
        queue_info[0].queueCount = 1;
        queue_info[0].pQueuePriorities = queue_priority;
        VkDeviceCreateInfo create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        create_info.queueCreateInfoCount =
            sizeof(queue_info) / sizeof(queue_info[0]);
        create_info.pQueueCreateInfos = queue_info;
        create_info.enabledExtensionCount = device_extension_count;
        create_info.ppEnabledExtensionNames = device_extensions;
        err = vkCreateDevice(g_PhysicalDevice, &create_info, g_allocator,
                             &g_Device);
        check_vk_result(err);
        vkGetDeviceQueue(g_Device, g_QueueFamily, 0, &g_Queue);
    }

    {
        VkDescriptorPoolSize pool_sizes[] = {
            {VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000},
            {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000},
            {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000},
            {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000}};
        VkDescriptorPoolCreateInfo pool_info = {};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        pool_info.maxSets = 1000 * IM_ARRAYSIZE(pool_sizes);
        pool_info.poolSizeCount = (uint32_t)IM_ARRAYSIZE(pool_sizes);
        pool_info.pPoolSizes = pool_sizes;
        err = vkCreateDescriptorPool(g_Device, &pool_info, g_allocator,
                                     &g_DescriptorPool);
        check_vk_result(err);
    }
}

static void setup_vulkan_window(ImGui_ImplVulkanH_Window *wd,
                                VkSurfaceKHR surface, int width, int height,
                                bool vsyncEnabled) {
    wd->Surface = surface;

    VkBool32 res;
    vkGetPhysicalDeviceSurfaceSupportKHR(g_PhysicalDevice, g_QueueFamily,
                                         wd->Surface, &res);
    if (res != VK_TRUE) {
        fprintf(stderr, "Error no WSI support on physical device 0\n");
        exit(-1);
    }

    const VkFormat requestSurfaceImageFormat[] = {
        VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM,
        VK_FORMAT_B8G8R8_UNORM, VK_FORMAT_R8G8B8_UNORM};
    const VkColorSpaceKHR requestSurfaceColorSpace =
        VK_COLORSPACE_SRGB_NONLINEAR_KHR;
    wd->SurfaceFormat = ImGui_ImplVulkanH_SelectSurfaceFormat(
        g_PhysicalDevice, wd->Surface, requestSurfaceImageFormat,
        (size_t)IM_ARRAYSIZE(requestSurfaceImageFormat),
        requestSurfaceColorSpace);

    VkPresentModeKHR present_modes[2];
    if (vsyncEnabled)
        present_modes[0] = VK_PRESENT_MODE_FIFO_KHR;  // V-Sync
    else {
        present_modes[0] = VK_PRESENT_MODE_MAILBOX_KHR;    // Triple buffering
        present_modes[1] = VK_PRESENT_MODE_IMMEDIATE_KHR;  // No waiting
    }

    wd->PresentMode = ImGui_ImplVulkanH_SelectPresentMode(
        g_PhysicalDevice, wd->Surface, &present_modes[0], vsyncEnabled ? 1 : 2);
    // printf("[vulkan] Selected PresentMode = %d\n", wd->PresentMode);

    IM_ASSERT(g_MinImageCount >= 2);
    ImGui_ImplVulkanH_CreateOrResizeWindow(
        g_Instance, g_PhysicalDevice, g_Device, wd, g_QueueFamily, g_allocator,
        width, height, g_MinImageCount, VK_IMAGE_USAGE_SAMPLED_BIT);
}

static void cleanup_vulkan() {
    vkDestroyDescriptorPool(g_Device, g_DescriptorPool, g_allocator);

#ifdef IMGUI_VULKAN_DEBUG_REPORT
    // Remove the debug report callback
    auto vkDestroyDebugReportCallbackEXT =
        (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(
            g_Instance, "vkDestroyDebugReportCallbackEXT");
    vkDestroyDebugReportCallbackEXT(g_Instance, g_DebugReport, g_allocator);
#endif  // IMGUI_VULKAN_DEBUG_REPORT

    vkDestroyDevice(g_Device, g_allocator);
    vkDestroyInstance(g_Instance, g_allocator);
}

static void cleanup_vulkan_window() {
    ImGui_ImplVulkanH_DestroyWindow(g_Instance, g_Device, &g_MainWindowData,
                                    g_allocator);
}

void frame_render(ImGui_ImplVulkanH_Window *wd, ImDrawData *draw_data) {
    VkResult err;

    VkSemaphore image_acquired_semaphore =
        wd->FrameSemaphores[wd->SemaphoreIndex].ImageAcquiredSemaphore;
    VkSemaphore render_complete_semaphore =
        wd->FrameSemaphores[wd->SemaphoreIndex].RenderCompleteSemaphore;
    err = vkAcquireNextImageKHR(g_Device, wd->Swapchain, UINT64_MAX,
                                image_acquired_semaphore, VK_NULL_HANDLE,
                                &wd->FrameIndex);
    if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR) {
        g_SwapChainRebuild = true;
        return;
    }
    check_vk_result(err);

    s_CurrentFrameIndex =
        (s_CurrentFrameIndex + 1) % g_MainWindowData.ImageCount;

    ImGui_ImplVulkanH_Frame *fd = &wd->Frames[wd->FrameIndex];
    {
        err = vkWaitForFences(
            g_Device, 1, &fd->Fence, VK_TRUE,
            UINT64_MAX);  // wait indefinitely instead of periodically checking
        check_vk_result(err);

        err = vkResetFences(g_Device, 1, &fd->Fence);
        check_vk_result(err);
    }

    {
        for (auto &func : s_ResourceFreeQueue[s_CurrentFrameIndex]) func();
        s_ResourceFreeQueue[s_CurrentFrameIndex].clear();
    }
    {
        auto &allocatedCommandBuffers =
            s_AllocatedCommandBuffers[wd->FrameIndex];
        if (allocatedCommandBuffers.size() > 0) {
            vkFreeCommandBuffers(g_Device, fd->CommandPool,
                                 (uint32_t)allocatedCommandBuffers.size(),
                                 allocatedCommandBuffers.data());
            allocatedCommandBuffers.clear();
        }

        err = vkResetCommandPool(g_Device, fd->CommandPool, 0);
        check_vk_result(err);
        VkCommandBufferBeginInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        err = vkBeginCommandBuffer(fd->CommandBuffer, &info);
        check_vk_result(err);
    }
    {
        VkRenderPassBeginInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        info.renderPass = wd->RenderPass;
        info.framebuffer = fd->Framebuffer;
        info.renderArea.extent.width = wd->Width;
        info.renderArea.extent.height = wd->Height;
        info.clearValueCount = 1;
        info.pClearValues = &wd->ClearValue;
        vkCmdBeginRenderPass(fd->CommandBuffer, &info,
                             VK_SUBPASS_CONTENTS_INLINE);
    }

    ImGui_ImplVulkan_RenderDrawData(draw_data, fd->CommandBuffer);

    vkCmdEndRenderPass(fd->CommandBuffer);
    {
        VkPipelineStageFlags wait_stage =
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

        std::vector<VkSemaphore> wait_semaphores;
        wait_semaphores.push_back(image_acquired_semaphore);

        std::vector<VkPipelineStageFlags> wait_stages;
        wait_stages.push_back(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

        if (s_instance) {
            wait_semaphores.insert(wait_semaphores.end(),
                                   s_instance->m_wait_semaphores.begin(),
                                   s_instance->m_wait_semaphores.end());
            for (size_t i = 0; i < s_instance->m_wait_semaphores.size(); ++i) {
                wait_stages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
            }
            s_instance->m_wait_semaphores.clear();
        }

        VkSubmitInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        info.waitSemaphoreCount = (uint32_t)wait_semaphores.size();
        info.pWaitSemaphores = wait_semaphores.data();
        info.pWaitDstStageMask = wait_stages.data();
        info.commandBufferCount = 1;
        info.pCommandBuffers = &fd->CommandBuffer;
        info.signalSemaphoreCount = 1;
        info.pSignalSemaphores = &render_complete_semaphore;

        err = vkEndCommandBuffer(fd->CommandBuffer);
        check_vk_result(err);
        err = vkQueueSubmit(g_Queue, 1, &info, fd->Fence);
        check_vk_result(err);
    }
}

static void frame_present(ImGui_ImplVulkanH_Window *wd) {
    if (g_SwapChainRebuild) return;
    VkSemaphore render_complete_semaphore =
        wd->FrameSemaphores[wd->SemaphoreIndex].RenderCompleteSemaphore;
    VkPresentInfoKHR info = {};
    info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    info.waitSemaphoreCount = 1;
    info.pWaitSemaphores = &render_complete_semaphore;
    info.swapchainCount = 1;
    info.pSwapchains = &wd->Swapchain;
    info.pImageIndices = &wd->FrameIndex;
    VkResult err = vkQueuePresentKHR(g_Queue, &info);
    if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR) {
        g_SwapChainRebuild = true;
        return;
    }
    check_vk_result(err);
    wd->SemaphoreIndex = (wd->SemaphoreIndex + 1) % wd->ImageCount;
}

static void glfw_error_callback(int error, const char *description) {
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

application::application(const specs &specification) : m_specs(specification) {
    logger::init();
    s_instance = this;

    init();
}

application::~application() {
    shutdown();

    s_instance = nullptr;
}

application &application::get() { return *s_instance; }

void application::init() {
    glfwSetErrorCallback(glfw_error_callback);

#ifdef __linux
    const char *forceX11 = getenv("FORCE_X11");
    if (forceX11 && strcmp(forceX11, "1") == 0) {
        std::cout << "FORCE_X11 is set to 1, forcing GLFW to use X11\n";
        glfwInitHint(GLFW_X11_XCB_VULKAN_SURFACE, GLFW_TRUE);
        glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11);
    }
#endif

    if (!glfwInit()) {
        std::cerr << "Could not initialize GLFW!\n";
        return;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_TRUE);

    m_window = glfwCreateWindow(m_specs.width, m_specs.height,
                                m_specs.name.c_str(), nullptr, nullptr);

    if (!glfwVulkanSupported()) {
        std::cerr << "GLFW: Vulkan not supported!\n";
        return;
    }
    uint32_t extensions_count = 0;
    const char **extensions =
        glfwGetRequiredInstanceExtensions(&extensions_count);
    setup_vulkan(extensions, extensions_count);

    VkSurfaceKHR surface;
    VkResult err =
        glfwCreateWindowSurface(g_Instance, m_window, g_allocator, &surface);
    check_vk_result(err);

    // Create Frame buffers
    int w, h;
    glfwGetFramebufferSize(m_window, &w, &h);
    ImGui_ImplVulkanH_Window *wd = &g_MainWindowData;
    setup_vulkan_window(wd, surface, w, h, m_vsync_enabled);

    s_AllocatedCommandBuffers.resize(wd->ImageCount);
    s_ResourceFreeQueue.resize(wd->ImageCount);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    bool supportsAbsoluteWindowPos = false;
    bool supportsDocking = false;

    // There must be a better way to do this
    const int platform = glfwGetPlatform();
    if (platform == GLFW_PLATFORM_WIN32) {
        supportsAbsoluteWindowPos = true;
        supportsDocking = true;
    } else if (platform == GLFW_PLATFORM_X11) {
        supportsAbsoluteWindowPos = true;
        supportsDocking = true;
    } else if (platform == GLFW_PLATFORM_WAYLAND) {
        supportsAbsoluteWindowPos = false;
        supportsDocking = false;
    }

    ImGuiIO &io = ImGui::GetIO();
    (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    if (supportsDocking) io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    if (supportsAbsoluteWindowPos)
        io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

    ImGui::StyleColorsDark();

    ImGuiStyle &style = ImGui::GetStyle();
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
        style.WindowRounding = 0.0f;
        style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }

    ImGui_ImplGlfw_InitForVulkan(m_window, true);
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = g_Instance;
    init_info.PhysicalDevice = g_PhysicalDevice;
    init_info.Device = g_Device;
    init_info.QueueFamily = g_QueueFamily;
    init_info.Queue = g_Queue;
    init_info.PipelineCache = g_PipelineCache;
    init_info.DescriptorPool = g_DescriptorPool;
    init_info.MinImageCount = g_MinImageCount;
    init_info.ImageCount = wd->ImageCount;
    init_info.Allocator = g_allocator;
    init_info.CheckVkResultFn = check_vk_result;
    init_info.PipelineInfoMain.Subpass = 0;
    init_info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    init_info.PipelineInfoMain.RenderPass = wd->RenderPass;
    ImGui_ImplVulkan_Init(&init_info);

    ImFontConfig fontConfig;
    fontConfig.FontDataOwnedByAtlas = false;
    ImFont *robotoFont = io.Fonts->AddFontFromMemoryTTF(
        (void *)g_RobotoRegular, sizeof(g_RobotoRegular), 16.0f, &fontConfig);
    io.FontDefault = robotoFont;

    {
        VkCommandPool command_pool = wd->Frames[wd->FrameIndex].CommandPool;
        VkCommandBuffer command_buffer =
            wd->Frames[wd->FrameIndex].CommandBuffer;

        err = vkResetCommandPool(g_Device, command_pool, 0);
        check_vk_result(err);
        VkCommandBufferBeginInfo begin_info = {};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        err = vkBeginCommandBuffer(command_buffer, &begin_info);
        check_vk_result(err);

        // ImGui_ImplVulkan_CreateFontsTexture();

        VkSubmitInfo end_info = {};
        end_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        end_info.commandBufferCount = 1;
        end_info.pCommandBuffers = &command_buffer;
        err = vkEndCommandBuffer(command_buffer);
        check_vk_result(err);
        err = vkQueueSubmit(g_Queue, 1, &end_info, VK_NULL_HANDLE);
        check_vk_result(err);

        err = vkDeviceWaitIdle(g_Device);
        check_vk_result(err);
    }
}

void application::shutdown() {
    for (auto &layer : m_layers) layer->on_detach();

    m_layers.clear();

    VkResult err = vkDeviceWaitIdle(g_Device);
    check_vk_result(err);

    for (auto &queue : s_ResourceFreeQueue) {
        for (auto &func : queue) func();
    }
    s_ResourceFreeQueue.clear();

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    cleanup_vulkan_window();
    cleanup_vulkan();

    glfwDestroyWindow(m_window);
    glfwTerminate();

    g_running = false;
}

void application::run() {
    m_running = true;

    ImGui_ImplVulkanH_Window *wd = &g_MainWindowData;
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    ImGuiIO &io = ImGui::GetIO();

    while (!glfwWindowShouldClose(m_window) && m_running) {
        glfwPollEvents();

        for (auto &layer : m_layers) layer->on_update(m_time_step);

        if (g_SwapChainRebuild) {
            int width, height;
            glfwGetFramebufferSize(m_window, &width, &height);
            if (width > 0 && height > 0) {
                ImGui_ImplVulkan_SetMinImageCount(g_MinImageCount);
                // ImGui_ImplVulkanH_CreateOrResizeWindow(g_Instance,
                // g_PhysicalDevice, g_Device, &g_MainWindowData, g_QueueFamily,
                // g_allocator, width, height, g_MinImageCount);
                setup_vulkan_window(&g_MainWindowData, g_MainWindowData.Surface,
                                    width, height, m_vsync_enabled);
                g_MainWindowData.FrameIndex = 0;

                s_AllocatedCommandBuffers.clear();
                s_AllocatedCommandBuffers.resize(g_MainWindowData.ImageCount);

                g_SwapChainRebuild = false;
            }
        }

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        {
            static ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_None;

            ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDocking;
            if (m_menubar_callback) window_flags |= ImGuiWindowFlags_MenuBar;

            const ImGuiViewport *viewport = ImGui::GetMainViewport();
            ImGui::SetNextWindowPos(viewport->WorkPos);
            ImGui::SetNextWindowSize(viewport->WorkSize);
            ImGui::SetNextWindowViewport(viewport->ID);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
            window_flags |= ImGuiWindowFlags_NoTitleBar |
                            ImGuiWindowFlags_NoCollapse |
                            ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
            window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus |
                            ImGuiWindowFlags_NoNavFocus;

            if (dockspace_flags & ImGuiDockNodeFlags_PassthruCentralNode)
                window_flags |= ImGuiWindowFlags_NoBackground;

            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding,
                                ImVec2(0.0f, 0.0f));
            ImGui::Begin("DockSpace Demo", nullptr, window_flags);
            ImGui::PopStyleVar();

            ImGui::PopStyleVar(2);

            ImGuiIO &io = ImGui::GetIO();
            if (io.ConfigFlags & ImGuiConfigFlags_DockingEnable) {
                ImGuiID dockspace_id = ImGui::GetID("VulkanAppDockspace");

                static bool first_time = true;
                if (first_time) {
                    first_time = false;

                    if (ImGui::DockBuilderGetNode(dockspace_id) == NULL) {
                        ImGui::DockBuilderRemoveNode(
                            dockspace_id);  // Clear out existing layout
                        ImGui::DockBuilderAddNode(
                            dockspace_id,
                            ImGuiDockNodeFlags_DockSpace);  // Add empty node
                        ImGui::DockBuilderSetNodeSize(dockspace_id,
                                                      viewport->Size);

                        ImGuiID dock_main_id = dockspace_id;
                        ImGuiID dock_id_left = ImGui::DockBuilderSplitNode(
                            dock_main_id, ImGuiDir_Left, 0.20f, nullptr,
                            &dock_main_id);
                        ImGuiID dock_id_right = ImGui::DockBuilderSplitNode(
                            dock_main_id, ImGuiDir_Right, 0.25f, nullptr,
                            &dock_main_id);
                        ImGuiID dock_id_right_down =
                            ImGui::DockBuilderSplitNode(
                                dock_id_right, ImGuiDir_Down, 0.50f, nullptr,
                                &dock_id_right);

                        ImGui::DockBuilderDockWindow("Hierarchy", dock_id_left);
                        ImGui::DockBuilderDockWindow("Inspector",
                                                     dock_id_right);
                        ImGui::DockBuilderDockWindow("Settings",
                                                     dock_id_right_down);
                        ImGui::DockBuilderDockWindow("Material settings",
                                                     dock_id_right_down);
                        ImGui::DockBuilderDockWindow("Light settings",
                                                     dock_id_right_down);
                        ImGui::DockBuilderDockWindow("Viewport", dock_main_id);
                        ImGui::DockBuilderFinish(dockspace_id);
                    }
                }

                ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f),
                                 dockspace_flags);
            }

            if (m_menubar_callback) {
                if (ImGui::BeginMenuBar()) {
                    m_menubar_callback();
                    ImGui::EndMenuBar();
                }
            }

            for (auto &layer : m_layers) layer->on_render();

            ImGui::End();
        }

        ImGui::Render();
        ImDrawData *main_draw_data = ImGui::GetDrawData();
        const bool main_is_minimized = (main_draw_data->DisplaySize.x <= 0.0f ||
                                        main_draw_data->DisplaySize.y <= 0.0f);
        wd->ClearValue.color.float32[0] = clear_color.x * clear_color.w;
        wd->ClearValue.color.float32[1] = clear_color.y * clear_color.w;
        wd->ClearValue.color.float32[2] = clear_color.z * clear_color.w;
        wd->ClearValue.color.float32[3] = clear_color.w;
        if (!main_is_minimized) frame_render(wd, main_draw_data);

        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
        }

        if (!main_is_minimized) frame_present(wd);

        float time = get_time();
        m_frame_time = time - m_last_frame_time;
        m_time_step = (0.0333f < m_frame_time) ? 0.0333f : m_frame_time;
        m_last_frame_time = time;
    }
}

void application::toggle_fullscreen() {
    GLFWmonitor *monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode *mode = glfwGetVideoMode(monitor);
    if (m_fullscreen) {
        glfwSetWindowMonitor(m_window, monitor, 0, 0, mode->width, mode->height,
                             mode->refreshRate);
    } else {
        glfwSetWindowMonitor(m_window, nullptr, 100, 100, m_specs.width,
                             m_specs.height, mode->refreshRate);
    }

    m_fullscreen = !m_fullscreen;
    g_SwapChainRebuild = true;
}

void application::toggle_vsync() {
    m_vsync_enabled = !m_vsync_enabled;
    g_SwapChainRebuild = true;
}

void application::close() { m_running = false; }

float application::get_time() { return static_cast<float>(glfwGetTime()); }

VkInstance application::get_instance() { return g_Instance; }

VkPhysicalDevice application::get_physical_device() { return g_PhysicalDevice; }

VkDevice application::get_device() { return g_Device; }

uint32_t application::get_graphics_queue_family_index() {
    return g_QueueFamily;
}

VkQueue application::get_graphics_queue() { return g_Queue; }

VkCommandBuffer application::get_command_buffer(bool begin) {
    ImGui_ImplVulkanH_Window *wd = &g_MainWindowData;

    // Use any command queue
    VkCommandPool command_pool = wd->Frames[wd->FrameIndex].CommandPool;

    VkCommandBufferAllocateInfo cmdBufAllocateInfo = {};
    cmdBufAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdBufAllocateInfo.commandPool = command_pool;
    cmdBufAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdBufAllocateInfo.commandBufferCount = 1;

    VkCommandBuffer &command_buffer =
        s_AllocatedCommandBuffers[wd->FrameIndex].emplace_back();
    auto err = vkAllocateCommandBuffers(g_Device, &cmdBufAllocateInfo,
                                        &command_buffer);

    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    err = vkBeginCommandBuffer(command_buffer, &begin_info);
    check_vk_result(err);

    return command_buffer;
}

VkCommandBuffer application::begin_single_time_commands() {
    VkCommandBuffer commandBuffer = get_command_buffer(true);
    return commandBuffer;
}

VkCommandBuffer application::submit_single_time_commands(
    VkCommandBuffer commandBuffer) {
    VkCommandBuffer command_buffer = commandBuffer;
    auto err = vkEndCommandBuffer(command_buffer);
    check_vk_result(err);

    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;

    err = vkQueueSubmit(g_Queue, 1, &submit_info, VK_NULL_HANDLE);
    check_vk_result(err);

    err = vkQueueWaitIdle(g_Queue);
    check_vk_result(err);

    return command_buffer;
}

void application::flush_command_buffer(VkCommandBuffer commandBuffer) {
    const uint64_t DEFAULT_FENCE_TIMEOUT = 100000000000;

    VkSubmitInfo end_info = {};
    end_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    end_info.commandBufferCount = 1;
    end_info.pCommandBuffers = &commandBuffer;
    auto err = vkEndCommandBuffer(commandBuffer);
    check_vk_result(err);

    // Create fence to ensure that the command buffer has finished executing
    VkFenceCreateInfo fenceCreateInfo = {};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.flags = 0;
    VkFence fence;
    err = vkCreateFence(g_Device, &fenceCreateInfo, nullptr, &fence);
    check_vk_result(err);

    err = vkQueueSubmit(g_Queue, 1, &end_info, fence);
    check_vk_result(err);

    err = vkWaitForFences(g_Device, 1, &fence, VK_TRUE, DEFAULT_FENCE_TIMEOUT);
    check_vk_result(err);

    vkDestroyFence(g_Device, fence, nullptr);
}

void application::submit_resource_free(std::function<void()> &&func) {
    s_ResourceFreeQueue[s_CurrentFrameIndex].emplace_back(func);
}

void application::add_wait_semaphore(VkSemaphore semaphore) {
    m_wait_semaphores.push_back(semaphore);
}