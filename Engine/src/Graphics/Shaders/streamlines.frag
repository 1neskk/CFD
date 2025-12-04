#version 450

layout (location = 0) in vec2 inUV;
layout (location = 0) out vec4 outColor;

layout (binding = 0) uniform sampler2D u_VelocityTexture;

// Pseudo-random noise
float rand(vec2 co){
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

void main() 
{
    // 1. Get Velocity
    vec2 vel_at_pixel = texture(u_VelocityTexture, inUV).xy;
    float speed = length(vel_at_pixel);

    // Optimization: Don't compute LIC for empty air or solids
    if (speed < 0.001) discard;

    // 2. LIC Algorithm (Simplified)
    ivec2 size = textureSize(u_VelocityTexture, 0);
    float step_len = 1.0 / float(max(size.x, size.y));
    float noise = rand(floor(inUV * vec2(size)));
    
    vec2 pos = inUV;
    float acc = noise;
    float w_sum = 1.0;
    
    // Integrate Forward (shorter steps for sharper lines)
    for(int i = 0; i < 10; i++) { 
        vec2 v = texture(u_VelocityTexture, pos).xy;
        if(length(v) < 0.001) break;
        pos += normalize(v) * step_len;
        acc += rand(floor(pos * vec2(size)));
        w_sum += 1.0;
    }
    
    // Normalize noise
    float intensity = acc / w_sum;
    
    // 3. Contrast & Thresholding
    // This makes the lines look like sharp "wires" instead of fuzzy noise
    float alpha = smoothstep(0.45, 0.55, intensity);
    
    // Fade out based on speed (stagnant air shouldn't have strong lines)
    alpha *= smoothstep(0.0, 0.02, speed);

    // Output: Pure White, with alpha determined by LIC
    outColor = vec4(1.0, 1.0, 1.0, alpha * 0.5); // 0.5 master opacity
}