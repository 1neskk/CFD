#version 450

layout (location = 0) in vec2 inUV;
layout (location = 0) out vec4 outColor;

layout (binding = 0) uniform sampler2D u_VelocityTexture; // Input: Velocity Field
layout (binding = 1) uniform sampler2D u_SolidTexture;    // Input: Solid Mask (R8_UNORM)

// Google's Turbo Colormap (Fast Polynomial Approximation)
// Input: t [0.0, 1.0]
vec3 turbo(float t) {
    const vec3 c0 = vec3(0.114089, 0.062866, 0.224861);
    const vec3 c1 = vec3(0.663470, 0.185305, 0.317349);
    const vec3 c2 = vec3(-0.200529, 1.194722, 0.018450);
    const vec3 c3 = vec3(0.026700, 0.050840, -0.363970);
    const vec3 c4 = vec3(0.060515, -0.005725, -0.074261);
    const vec3 c5 = vec3(-0.012140, 0.006574, 0.017982);
    return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))));
}

void main() 
{
    // 1. Check for Solid Boundaries first
    float solid = texture(u_SolidTexture, inUV).r;
    
    // If this pixel is part of the car/wall, draw it flat Grey
    if (solid > 0.1) {
        outColor = vec4(0.2, 0.2, 0.2, 1.0); // Slate Grey
        return;
    }

    // 2. Visualize Fluid Velocity
    vec2 vel = texture(u_VelocityTexture, inUV).xy;
    float speed = length(vel);
    
    // Scale speed to 0.0-1.0 range for the colormap
    // You must tune 'max_speed' to match your simulation's typical values
    // For LBM, max speed is usually around 0.1 to 0.2
    float max_speed = 0.15; 
    float t = clamp(speed / max_speed, 0.0, 1.0);
    
    // Apply Turbo Colormap
    vec3 fluidColor = turbo(t);

    outColor = vec4(fluidColor, 1.0);
}