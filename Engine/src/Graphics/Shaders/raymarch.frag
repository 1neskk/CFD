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
    vec3 sim_dimensions;
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

    vec4 target = camera.inverse_projection * vec4(uv.x, uv.y, 1.0, 1.0);
    vec3 rayDir = normalize(vec3(camera.inverse_view * vec4(normalize(target.xyz / target.w), 0.0)));
    vec3 rayOrigin = camera.position;
    
    vec3 rd = rayDir;
    vec3 camPos = rayOrigin;

    vec3 boxMax = vec3(1.0);
    vec3 boxMin = vec3(0.0);
    
    vec2 tHit = intersectBox(camPos, rd, boxMin, boxMax);
    
    // Miss
    if (tHit.x > tHit.y) {
        outColor = vec4(0.1, 0.1, 0.1, 1.0);
        return;
    }
    
    float tStart = max(tHit.x, 0.0);
    float tEnd = tHit.y;
    
    // Raymarching for Isosurface
    vec3 rayPos = camPos + tStart * rd;
    vec4 color = vec4(0.0);
    
    int steps = 128; 
    float stepSize = (tEnd - tStart) / float(steps);
    
    float isoSpeed = 0.01; 
    float prevSpeed = 0.0;
    
    if (tStart == 0.0) {
        prevSpeed = length(texture(velocityTexture, camPos).xyz);
    } else {
        prevSpeed = length(texture(velocityTexture, rayPos).xyz);
    }

    vec3 topColor = vec3(0.2, 0.2, 0.3);
    vec3 bottomColor = vec3(0.1, 0.1, 0.1);
    float t = 0.5 * (rd.y + 1.0);
    vec3 bgColor = mix(bottomColor, topColor, t);
    
    bool hit = false;
    vec3 hitPos = vec3(0.0);
    vec3 surfaceColor = vec3(0.0);

    for (int i = 0; i < steps; i++) {
        vec3 pos = rayPos;
        
        // Check solid
        float solid = texture(solidTexture, pos).r;
        if (solid > 0.5) {
            // Solid surface
            color = vec4(0.4, 0.4, 0.4, 1.0);
            hit = true;
            break; 
        }
        
        // Sample velocity magnitude
        vec4 vel = texture(velocityTexture, pos);
        float speed = length(vel.xyz);
        
        // Check for isosurface crossing
        if ((prevSpeed < isoSpeed && speed>= isoSpeed) || (prevSpeed > isoSpeed && speed<= isoSpeed)) {
            // Linear interpolation
            float tOffset = (isoSpeed - prevSpeed) / (speed - prevSpeed);
            hitPos = pos - rd * stepSize * (1.0 - tOffset);
            
            vec3 normal = normalize(vel.xyz); 
            
            float diffuse = max(dot(normal, -rd), 0.0);
            vec3 baseColor = turbo(speed * 5.0);
            
            surfaceColor = baseColor * (0.5 + 0.5 * diffuse);
            color = vec4(surfaceColor, 1.0);
            hit = true;
            break;
        }
        
        prevSpeed = speed;
        rayPos += rd * stepSize;
    }
    
    if (hit) {
        outColor = color;
    } else {
        outColor = vec4(bgColor, 1.0);
    }
}
