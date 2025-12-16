#pragma once

#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

#include "cuda_buffer.cuh"

struct device_camera {
    alignas(16) glm::vec3 position;
    alignas(16) glm::vec3 direction;
    alignas(16) uint32_t width;
    uint32_t height;

    alignas(16) glm::mat4 inverse_view;
    alignas(16) glm::mat4 inverse_projection;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 projection;
    alignas(16) glm::vec3 sim_dimensions;
};

class camera {
   public:
    camera() = default;
    camera(float fov, float near_clip, float far_clip);
    camera(float fov, float near_clip, float far_clip, glm::vec3 position,
           glm::vec3 direction);

    camera& operator=(const camera& other) {
        m_projection = other.m_projection;
        m_view = other.m_view;
        m_inverse_projection = other.m_inverse_projection;
        m_inverse_view = other.m_inverse_view;
        m_position = other.m_position;
        m_direction = other.m_direction;
        m_fov = other.m_fov;
        m_near_clip = other.m_near_clip;
        m_far_clip = other.m_far_clip;
        m_last_mouse_pos = other.m_last_mouse_pos;
        m_width = other.m_width;
        m_height = other.m_height;
        m_view_dirty = other.m_view_dirty;
        m_projection_dirty = other.m_projection_dirty;
        return *this;
    }

    bool on_update(float dt);
    void resize(uint32_t width, uint32_t height);

    void allocate_device_resources(device_camera& device_camera) const;
    static void free_device_resources(device_camera& device_camera);

    const glm::mat4& get_projection() const { return m_projection; }
    const glm::mat4& get_view() const { return m_view; }
    const glm::mat4& get_inverse_projection() const {
        return m_inverse_projection;
    }
    const glm::mat4& get_inverse_view() const { return m_inverse_view; }
    const glm::vec3& get_position() const { return m_position; }
    const glm::vec3& get_direction() const { return m_direction; }
    const float& get_fov() const { return m_fov; }

    void set_position(const glm::vec3& position) {
        m_position = position;
        m_view_dirty = true;
    }

    void set_direction(const glm::vec3& direction) {
        m_direction = direction;
        m_view_dirty = true;
    }

    void set_fov(float fov) {
        m_fov = fov;
        m_projection_dirty = true;
    }

    static float get_rotation_speed();

   private:
    void update_projection();
    void update_view();

   private:
    glm::mat4 m_projection{1.0f}, m_view{1.0f};
    glm::mat4 m_inverse_projection{1.0f}, m_inverse_view{1.0f};

    glm::vec3 m_position{0.0f}, m_direction{0.0f};

    float m_fov = 45.0f;
    float m_near_clip = 0.1f;
    float m_far_clip = 100.0f;

    glm::vec2 m_last_mouse_pos{0.0f};
    uint32_t m_width = 0, m_height = 0;

    bool m_view_dirty = true;
    bool m_projection_dirty = true;
};