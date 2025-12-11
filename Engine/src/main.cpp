#include "application.h"
#include "logger.h"
#include "entry_point.h"
#include "image.h"
#include "timer.h"
#include "Graphics/renderer.h"
#include "Physics/lbm_solver.cuh"

class cfd final : public layer {
public:
    cfd() {
        m_renderer = std::make_unique<renderer>(400, 100, 100);
        m_solver = std::make_unique<lbm_solver>(400, 100, 100);
        
        m_solver->register_external_velocity(m_renderer->get_velocity_fd(), 400 * 100 * 100 * sizeof(float) * 4);
        m_solver->register_external_solid(m_renderer->get_solid_fd(), 400 * 100 * 100 * sizeof(uint8_t));
        
        m_solver->init();
        
        // pos, w, h, d;
        lbm_solver::Rect rect = {150, 40, 40, 80, 20, 20};
        m_solver->add_solid(rect);

#ifdef _DEBUG
        LOG_INFO("CFD Layer initialized");
#endif
    }

    virtual void on_update(const float ts) override {
        m_solver->step();
        m_renderer->update_sim_data();
        m_renderer->update_camera(ts);
    }

    virtual void on_render() override {
        const auto& io = ImGui::GetIO();
        
        style::theme();

        ImGui::Begin("Renderer Settings");
        ImGui::Text("Last Render Time: %.3fms", m_last_render_time);
        ImGui::Text("App Frame Time: %.3fms (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
        
        bool vsync = application::get().is_vsync_enabled();
        if (ImGui::Checkbox("VSync", &vsync)) {
            application::get().toggle_vsync();
        }
        ImGui::End();

        ImGui::Begin("CFD Settings");
        ImGui::SliderFloat("tau", &m_solver->get_settings().tau, 0.1f, 1.0f);
        ImGui::SliderFloat("inlet velocity", &m_solver->get_settings().inlet_velocity, 0.0f, 1.0f);
        ImGui::End();

        ImGui::Begin("Viewport", nullptr,
                    ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
                    ImGuiWindowFlags_NoTitleBar |
                    ImGuiWindowFlags_NoScrollbar);
        
        ImVec2 viewport_panel_size = ImGui::GetContentRegionAvail();
        m_viewport_width = viewport_panel_size.x;
        m_viewport_height = viewport_panel_size.y;

        auto image = m_renderer->get_output_image();
        if (image) {
            ImGui::Image(image->get_descriptor_set(), 
                         {static_cast<float>(image->get_width()),
                          static_cast<float>(image->get_height())},
                          ImVec2(0, 1), ImVec2(1, 0));
        }
        
        ImGui::End();

        render();
    }

    void render() {
        timer timer;
        m_renderer->resize(m_viewport_width, m_viewport_height);
        m_renderer->render();
        m_last_render_time = timer.elapsed_ms();
    }

private:
    std::unique_ptr<lbm_solver> m_solver;
    std::unique_ptr<renderer> m_renderer;

    int m_viewport_width = 0, m_viewport_height = 0;
    float m_last_render_time = 0.0f;
};

application* create_application(int argc, char** argv) {
    specs spec;
    spec.name = "CFD Engine";

    auto app = new application(spec);
    app->push_layer<cfd>();
    return app;
}
