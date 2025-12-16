#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec3 fragNormal;
layout(location = 1) out float fragSpeed;

layout(binding = 0) uniform sampler3D velocityTexture;

layout(binding = 2) uniform Camera {
    vec3 position;
    vec3 direction;
    uint width;
    uint height;
    mat4 inverse_view;
    mat4 inverse_projection;
    mat4 view;
    mat4 projection;
} camera;

layout(push_constant) uniform PushConstants {
    uvec3 gridSize;
    float scale;
} push;

mat3 align_vectors(vec3 from, vec3 to) {
    vec3 v = cross(from, to);
    float c = dot(from, to);
    float k = 1.0 / (1.0 + c);

    return mat3(
        v.x * v.x * k + c,     v.y * v.x * k - v.z,   v.z * v.x * k + v.y,
        v.x * v.y * k + v.z,   v.y * v.y * k + c,     v.z * v.y * k - v.x,
        v.x * v.z * k - v.y,   v.y * v.z * k + v.x,   v.z * v.z * k + c
    );
}

void main() {
    uint index = gl_InstanceIndex;
    
    uint z = index / (push.gridSize.x * push.gridSize.y);
    uint rem = index % (push.gridSize.x * push.gridSize.y);
    uint y = rem / push.gridSize.x;
    uint x = rem % push.gridSize.x;
    
    vec3 gridPos = vec3(float(x), float(y), float(z));
    vec3 uvw = (gridPos + 0.5) / vec3(push.gridSize);
    
    vec4 vel = texture(velocityTexture, uvw);
    float speed = length(vel.xyz);
    
    fragSpeed = speed;
    fragNormal = inNormal;
    
    if (speed < 0.001) {
        gl_Position = vec4(0.0, 0.0, 0.0, 0.0);
        return;
    }
    
    vec3 dir = normalize(vel.xyz);
    vec3 up = vec3(0.0, 1.0, 0.0);
    
    // Rotate arrow
    mat3 rot = mat3(1.0);
    if (abs(dot(up, dir)) < 0.999) {
        rot = align_vectors(up, dir);
    } else if (dot(up, dir) < -0.999) {
        rot = mat3(-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0); // Flip
    }
    
    fragNormal = rot * inNormal;
    
    vec3 pos = rot * (inPosition * push.scale * 20.0) + uvw; // Map to [0, 1] space
    gl_Position = camera.projection * camera.view * vec4(pos, 1.0);
}
