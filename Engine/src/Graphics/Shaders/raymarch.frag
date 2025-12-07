#version 450

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform sampler3D velocityTexture;
layout(binding = 1) uniform sampler3D solidTexture;

layout(binding = 2) uniform Camera {
    vec3 position;
    vec3 direction;
    uint width;
    uint height;
    mat4 inverse_view;
    mat4 inverse_projection;
} camera;

// Simple Turbo colormap
vec3 turbo(float t) {
    const vec3 c0 = vec3(0.114080, 0.062872, 0.224851);
    const vec3 c1 = vec3(0.663469, 0.185305, 0.114478);
    const vec3 c2 = vec3(0.276914, 0.814836, 0.073329);
    const vec3 c3 = vec3(0.124352, -0.674032, 0.134589);
    const vec3 c4 = vec3(0.011794, 0.596631, -0.005132);
    const vec3 c5 = vec3(-0.012116, -0.238828, 0.002615);
    const vec3 c6 = vec3(0.003642, 0.043759, -0.000538);
    const vec3 c7 = vec3(-0.000491, -0.004561, 0.000066);

    return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * (c6 + t * c7))))));
}

// Ray-Box Intersection
// Returns (tNear, tFar)
vec2 intersectBox(vec3 ro, vec3 rd, vec3 boxMin, vec3 boxMax) {
    vec3 tMin = (boxMin - ro) / rd;
    vec3 tMax = (boxMax - ro) / rd;
    vec3 t1 = min(tMin, tMax);
    vec3 t2 = max(tMin, tMax);
    float tNear = max(max(t1.x, t1.y), t1.z);
    float tFar = min(min(t2.x, t2.y), t2.z);
    return vec2(tNear, tFar);
}

void main() {
    // Map UV to [-1, 1]
    vec2 uv = inUV * 2.0 - 1.0;

    // Calculate ray origin and direction from inverse matrices
    vec4 target = camera.inverse_projection * vec4(uv.x, uv.y, 1.0, 1.0);
    vec3 rayDir = normalize(vec3(camera.inverse_view * vec4(normalize(target.xyz / target.w), 0.0)));
    vec3 rayOrigin = camera.position;
    
    // Ray direction
    vec3 rd = rayDir;
    vec3 camPos = rayOrigin;
    
    // Volume bounds [0, 1]^3
    vec3 boxMin = vec3(0.0);
    vec3 boxMax = vec3(1.0);
    
    vec2 tHit = intersectBox(camPos, rd, boxMin, boxMax);
    
    if (tHit.x > tHit.y) {
        // Miss
        outColor = vec4(0.1, 0.1, 0.1, 1.0); // Dark background
        return;
    }
    
    // Clamp start to 0 if inside box
    float tStart = max(tHit.x, 0.0);
    float tEnd = tHit.y;
    
    // Raymarching
    vec3 rayPos = camPos + tStart * rd;
    vec4 color = vec4(0.0);
    
    int steps = 64;
    float stepSize = (tEnd - tStart) / float(steps);
    
    for (int i = 0; i < steps; i++) {
        if (color.a >= 0.99) break;
        
        // Sample textures
        // We are inside [0, 1] box, so rayPos is UVW
        vec3 pos = rayPos;
        
        // Check solid
        float solid = texture(solidTexture, pos).r;
        if (solid > 0.5) {
            // Solid surface
            vec4 solidColor = vec4(0.4, 0.4, 0.4, 1.0); // Grey
            color = color + solidColor * (1.0 - color.a);
            break; // Stop raymarching
        }
        
        // Sample velocity magnitude
        vec4 vel = texture(velocityTexture, pos);
        float speed = length(vel.xyz);
        
        // Transfer function
        // Map speed to color
        vec3 sampleColor = turbo(clamp(speed * 5.0, 0.0, 1.0)); // Scale speed for visibility
        float alpha = clamp(speed * 2.0, 0.0, 0.1); // Low opacity for transparent volume
        
        // Accumulate
        color.rgb += sampleColor * alpha * (1.0 - color.a);
        color.a += alpha * (1.0 - color.a);
        
        rayPos += rd * stepSize;
    }
    
    // Background blending
    vec3 bgColor = vec3(0.1, 0.1, 0.1);
    outColor = vec4(color.rgb + bgColor * (1.0 - color.a), 1.0);
}
