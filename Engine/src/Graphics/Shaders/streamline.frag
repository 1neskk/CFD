#version 450

layout(location = 0) in float fragSpeed;

layout(location = 0) out vec4 outColor;

// Turbo colormap
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

void main() {
    vec3 color = turbo(clamp(fragSpeed * 5.0, 0.0, 1.0));
    outColor = vec4(color, 1.0);
}
