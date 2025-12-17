#pragma once

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_vulkan.h>
#include <vulkan/vulkan.h>

#include <functional>
#include <memory>
#include <string>
#include <vector>

class layer {
   public:
    virtual ~layer() = default;

    virtual void on_attach() {}
    virtual void on_detach() {}

    virtual void on_update(float ts) {}
    virtual void on_render() {}
};

void check_vk_result(VkResult result);

struct GLFWwindow;

struct specs {
    std::string name = "Vulkan Application";
    int width = 1600, height = 900;
    int window_pos_x = 0, window_pos_y = 300;
};

class application {
   public:
    application(const specs &specification = specs());
    ~application();

    static application &get();

    void run();
    void set_menubar_callback(const std::function<void()> &callback) {
        m_menubar_callback = callback;
    }

    template <typename T>
    void push_layer() {
        static_assert(std::is_base_of<layer, T>::value,
                      "T must derive from layer");
        m_layers.emplace_back(std::make_shared<T>())->on_attach();
    }

    void push_layer(const std::shared_ptr<layer> &layer) {
        m_layers.emplace_back(layer);
        layer->on_attach();
    }

    template <typename T>
    T *get_layer() {
        for (const auto &layer : m_layers) {
            if (typeid(*layer.get()) == typeid(T))
                return static_cast<T *>(layer.get());
        }
        return nullptr;
    }

    void pop_layer() {
        if (!m_layers.empty()) {
            m_layers.back()->on_detach();
            m_layers.pop_back();
        }
    }

    void close();
    float get_time();

    void toggle_fullscreen();
    void toggle_vsync();
    bool is_vsync_enabled() const { return m_vsync_enabled; }

    GLFWwindow *get_window() { return m_window; }

    static VkInstance get_instance();
    static VkPhysicalDevice get_physical_device();
    static VkDevice get_device();
    static uint32_t get_graphics_queue_family_index();
    static VkQueue get_graphics_queue();

    static VkCommandBuffer begin_single_time_commands();
    static VkCommandBuffer submit_single_time_commands(
        VkCommandBuffer command_buffer);

    static VkCommandBuffer get_command_buffer();
    static void flush_command_buffer(VkCommandBuffer command_buffer);

    static void submit_resource_free(std::function<void()> &&func);

    void add_wait_semaphore(VkSemaphore semaphore);
    friend void frame_render(ImGui_ImplVulkanH_Window *wd,
                             ImDrawData *draw_data);

   private:
    void init();
    void shutdown();

   private:
    specs m_specs;
    GLFWwindow *m_window = nullptr;
    bool m_running = true;
    bool m_vsync_enabled = true;
    bool m_fullscreen = true;
    GLFWmonitor *m_monitor = nullptr;
    const GLFWvidmode *m_video_mode = nullptr;

    float m_time_step = 0.0f;
    float m_frame_time = 0.0f;
    float m_last_frame_time = 0.0f;

    std::vector<std::shared_ptr<layer>> m_layers;
    std::function<void()> m_menubar_callback;

    std::vector<VkSemaphore> m_wait_semaphores;
};

application *create_application(int argc, char **argv);
