#include <cuda_runtime.h>
#include <limits>
#include <cstring>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>

#include "camera.h"
#include "Input/input.h"

camera::camera(float fov, float near_clip, float far_clip)
    : m_fov(fov), m_near_clip(near_clip), m_far_clip(far_clip) {
    m_direction = glm::vec3(0.0f, 0.0f, -1.0f);
    m_position = glm::vec3(0.0f, 0.0f, 3.0f);
    update_view();
    update_projection();
}

camera::camera(float fov, float near_clip, float far_clip, glm::vec3 position,
               glm::vec3 direction)
    : m_position(position),
      m_direction(direction),
      m_fov(fov),
      m_near_clip(near_clip),
      m_far_clip(far_clip) {
    update_view();
    update_projection();
}

bool camera::on_update(float dt) {
    glm::vec2 mouse_pos = input::input::get_mouse_position();
    glm::vec2 mouse_delta = (mouse_pos - m_last_mouse_pos) * 0.002f;
    m_last_mouse_pos = mouse_pos;

    if (!input::input::is_mouse_button_pressed(input::mouse_button::right)) {
        input::input::set_cursor_mode(input::cursor_mode::normal);
        return false;
    }
    
    input::input::set_cursor_mode(input::cursor_mode::locked);
    bool moved = false;

    constexpr glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 right = glm::cross(m_direction, up);

    float speed = 5.0f;

    if (input::input::is_key_pressed(input::key::w)) {
        m_position += m_direction * speed * dt;
        m_view_dirty = true;
        moved = true;
    } else if (input::input::is_key_pressed(input::key::s)) {
        m_position -= m_direction * speed * dt;
        m_view_dirty = true;
        moved = true;
    }

    if (input::input::is_key_pressed(input::key::a)) {
        m_position -= right * speed * dt;
        m_view_dirty = true;
        moved = true;
    } else if (input::input::is_key_pressed(input::key::d)) {
        m_position += right * speed * dt;
        m_view_dirty = true;
        moved = true;
    }

    if (input::input::is_key_pressed(input::key::q)) {
        m_position -= up * speed * dt;
        m_view_dirty = true;
        moved = true;
    } else if (input::input::is_key_pressed(input::key::e)) {
        m_position += up * speed * dt;
        m_view_dirty = true;
        moved = true;
    }

    float yaw_debug, pitch_debug;

    if (mouse_delta.x != 0.0f || mouse_delta.y != 0.0f) {
        float yaw_delta = mouse_delta.x * get_rotation_speed();
        float pitch_delta = mouse_delta.y * get_rotation_speed();

        yaw_debug = yaw_delta;
        pitch_debug = pitch_delta;

        glm::quat orientation = glm::normalize(glm::cross(glm::angleAxis(-pitch_delta, right),
                                glm::angleAxis(-yaw_delta, glm::vec3(0.0f, 1.0f, 0.0f))));

        m_direction = glm::rotate(orientation, m_direction);
        m_view_dirty = true;
        moved = true;
    }

    if (m_view_dirty) {
        update_view();
        update_projection();
    }
    return moved;
}

void camera::resize(uint32_t width, uint32_t height) {
    if (width == 0 || height == 0) {
        LOG_ERROR("Camera width and height cannot be zero!");
        return;
    }

    if (m_width == width && m_height == height) return;
    
    m_width = width;
    m_height = height;

    m_projection_dirty = true;
    update_projection();
}

void camera::update_projection() {
    if (m_projection_dirty) {
        if (m_width == 0 || m_height == 0)
            return;
        
        float aspect_ratio = static_cast<float>(m_width) / static_cast<float>(m_height);
        if (aspect_ratio <= std::numeric_limits<float>::epsilon()) {
            LOG_ERROR("Invalid camera aspect ratio: %f", aspect_ratio);
            return;
        }
        m_projection = glm::perspective(glm::radians(m_fov), aspect_ratio, m_near_clip, m_far_clip);
        m_inverse_projection = glm::inverse(m_projection);
        m_projection_dirty = false;
    }
}

void camera::update_view() {
    if (m_view_dirty) {
        m_view = glm::lookAt(m_position, m_position + m_direction, glm::vec3(0.0f, 1.0f, 0.0f));
        m_inverse_view = glm::inverse(m_view);
        m_view_dirty = false;
    }
}

void camera::allocate_device_resources(device_camera& device_camera) const {
    std::memset(&device_camera, 0, sizeof(device_camera));
    device_camera.position = m_position;
    device_camera.direction = m_direction;
    device_camera.width = m_width;
    device_camera.height = m_height;
    device_camera.inverse_view = m_inverse_view;
    device_camera.inverse_projection = m_inverse_projection;
}

void camera::free_device_resources(device_camera& /*device_camera*/) {
    // No dynamic allocation on device for now
}

float camera::get_rotation_speed() { return 0.3f; }
