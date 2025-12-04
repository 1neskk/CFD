#version 450

layout (location = 0) in vec2 inUV;
layout (location = 0) out vec4 outColor;

layout (binding = 0) uniform sampler2D u_DensityTexture;
layout (binding = 1) uniform sampler2D u_VelocityTexture;

// Turbo Colormap
// Adapted from Google's Turbo Colormap
vec3 turbo(float t) {
    const vec3 c0 = vec3(0.114089010972696, 0.06286598379113948, 0.22486109403887652);
    const vec3 c1 = vec3(0.6634697353363559, 0.18530514125586125, 0.31734879019313924);
    const vec3 c2 = vec3(-0.20052935582900175, 1.194721683979106, 0.018450369853293666);
    const vec3 c3 = vec3(0.026700039516330038, 0.0508404113385098, -0.3639702330887661);
    const vec3 c4 = vec3(0.060515439468528084, -0.005724579634221525, -0.07426117005978122);
    const vec3 c5 = vec3(-0.012140472895451977, 0.006574477192627562, 0.017982101901821736);

    return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))));
}

// Pseudo-random noise
float rand(vec2 co){
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

// Line Integral Convolution (LIC)
float lic(vec2 uv) {
    vec2 size = textureSize(u_VelocityTexture, 0);
    float L = 10.0; // Integration length
    float step = 1.0 / max(size.x, size.y);
    
    float noise = rand(floor(uv * size)); // Pixelated noise
    float acc = noise;
    float w_sum = 1.0;

    // Forward integration
    vec2 curr_uv = uv;
    for(int i = 0; i < 10; i++) {
        vec2 vel = texture(u_VelocityTexture, curr_uv).rg;
        if(length(vel) < 0.0001) break;
        
        vel = normalize(vel);
        curr_uv += vel * step;
        
        if(curr_uv.x < 0.0 || curr_uv.x > 1.0 || curr_uv.y < 0.0 || curr_uv.y > 1.0) break;

        float w = 1.0; // Uniform weighting for now
        acc += rand(floor(curr_uv * size)) * w;
        w_sum += w;
    }

    // Backward integration
    curr_uv = uv;
    for(int i = 0; i < 10; i++) {
        vec2 vel = texture(u_VelocityTexture, curr_uv).rg;
        if(length(vel) < 0.0001) break;
        
        vel = normalize(vel);
        curr_uv -= vel * step; // Subtract velocity
        
        if(curr_uv.x < 0.0 || curr_uv.x > 1.0 || curr_uv.y < 0.0 || curr_uv.y > 1.0) break;

        float w = 1.0;
        acc += rand(floor(curr_uv * size)) * w;
        w_sum += w;
    }

    return acc / w_sum;
}

void main() 
{
    float density = texture(u_DensityTexture, inUV).r;
    
    // Normalize density for visualization (assuming range around 1.0)
    // Adjust these bounds based on simulation stability
    float t = clamp((density - 0.95) * 10.0, 0.0, 1.0);
    
    vec3 heat = turbo(t);
    
    float stream = lic(inUV);
    
    // Blend heatmap and streamlines
    // Darken the heatmap where streamlines are dark
    vec3 finalColor = heat * (0.5 + 0.5 * stream);

    outColor = vec4(finalColor, 1.0);
}