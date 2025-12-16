#version 450

layout(location = 0) in vec4 inPosition; // xyz = pos, w = speed

layout(location = 0) out float fragSpeed;

layout(binding = 0) uniform Camera {
    vec3 position;
    vec3 direction;
    uint width;
    uint height;
    mat4 inverse_view;
    mat4 inverse_projection;
    mat4 view;
    mat4 projection;
} camera;

void main() {
    fragSpeed = inPosition.w;
    gl_Position = camera.projection * camera.view * vec4(inPosition.xyz, 1.0);
}
