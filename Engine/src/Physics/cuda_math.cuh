#pragma once
#include <cuda_runtime.h>

namespace cuda_math {
struct vec2 {
    float x, y;

    __device__ __forceinline__ constexpr vec2() = default;
    __device__ __forceinline__ constexpr vec2(float s) : x(s), y(s) {}
    __device__ __forceinline__ constexpr vec2(float xx, float yy)
        : x(xx), y(yy) {}
    __device__ __forceinline__ vec2(float2 v) : x(v.x), y(v.y) {}
    __device__ __forceinline__ operator float2() const {
        return make_float2(x, y);
    }
};

struct vec3 {
    float x, y, z;

    __device__ __forceinline__ constexpr vec3() = default;
    __device__ __forceinline__ constexpr vec3(float s) : x(s), y(s), z(s) {}
    __device__ __forceinline__ constexpr vec3(float xx, float yy, float zz)
        : x(xx), y(yy), z(zz) {}
    //__device__ __forceinline__ constexpr vec3(vec4 v) : x(v.x), y(v.y), z(v.z)
    //{}
    __device__ __forceinline__ vec3(float3 v) : x(v.x), y(v.y), z(v.z) {}
    __device__ __forceinline__ operator float3() const {
        return make_float3(x, y, z);
    }

    __device__ __forceinline__ vec3 operator-() const {
        return vec3(-x, -y, -z);
    }
};

struct vec4 {
    float x, y, z, w;

    __device__ __forceinline__ constexpr vec4() = default;
    __device__ __forceinline__ constexpr vec4(float s)
        : x(s), y(s), z(s), w(s) {}
    __device__ __forceinline__ constexpr vec4(float xx, float yy, float zz,
                                              float ww)
        : x(xx), y(yy), z(zz), w(ww) {}
    __device__ __forceinline__ constexpr vec4(vec3 xyz, float ww)
        : x(xyz.x), y(xyz.y), z(xyz.z), w(ww) {}
    __device__ __forceinline__ vec4(float4 v)
        : x(v.x), y(v.y), z(v.z), w(v.w) {}
    __device__ __forceinline__ operator float4() const {
        return make_float4(x, y, z, w);
    }

    __device__ __forceinline__ vec3 xyz() const { return vec3(x, y, z); }
};

__device__ __forceinline__ float2 operator+(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}
__device__ __forceinline__ float2 operator-(float2 a, float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}
__device__ __forceinline__ float2 operator*(float2 a, float s) {
    return make_float2(a.x * s, a.y * s);
}
__device__ __forceinline__ float2 operator*(float s, float2 a) { return a * s; }
__device__ __forceinline__ float2 operator/(float2 a, float s) {
    float inv = 1.0f / s;
    return a * inv;
}

__device__ __forceinline__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ __forceinline__ float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ __forceinline__ float3 operator*(float3 a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}
__device__ __forceinline__ float3 operator*(float s, float3 a) { return a * s; }
__device__ __forceinline__ float3 operator/(float3 a, float s) {
    float inv = 1.0f / s;
    return a * inv;
}
__device__ __forceinline__ float4 operator+(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
__device__ __forceinline__ float4 operator-(float4 a, float4 b) {
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
__device__ __forceinline__ float4 operator*(float4 a, float s) {
    return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}
__device__ __forceinline__ float4 operator*(float s, float4 a) { return a * s; }
__device__ __forceinline__ float4 operator/(float4 a, float s) {
    float inv = 1.0f / s;
    return a * inv;
}

// In-place
__device__ __forceinline__ float4& operator+=(float4& a, float4 b) {
    a = a + b;
    return a;
}
__device__ __forceinline__ float4& operator-=(float4& a, float4 b) {
    a = a - b;
    return a;
}
__device__ __forceinline__ float4& operator*=(float4& a, float s) {
    a = a * s;
    return a;
}

__device__ __forceinline__ float4 f4min(float4 a, float4 b) {
    return make_float4(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z),
                       fminf(a.w, b.w));
}
__device__ __forceinline__ float4 f4max(float4 a, float4 b) {
    return make_float4(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z),
                       fmaxf(a.w, b.w));
}
__device__ __forceinline__ float4 sqrt(float4 v) {
    return make_float4(sqrtf(v.x), sqrtf(v.y), sqrtf(v.z), sqrtf(v.w));
}
__device__ __forceinline__ float4 saturate(float4 v) {
    return f4min(f4max(v, make_float4(0.0f, 0.0f, 0.0f, 0.0f)),
                 make_float4(1.0f, 1.0f, 1.0f, 1.0f));
}

__device__ __forceinline__ float dot(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
__device__ __forceinline__ float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ __forceinline__ float3 cross(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                       a.x * b.y - a.y * b.x);
}

__device__ __forceinline__ float length_sq(float4 v) { return dot(v, v); }
__device__ __forceinline__ float length_sq(float3 v) { return dot(v, v); }
__device__ __forceinline__ float length(float4 v) { return sqrtf(dot(v, v)); }
__device__ __forceinline__ float length(float3 v) { return sqrtf(dot(v, v)); }

__device__ __forceinline__ float4 normalize(float4 v) {
    return v * rsqrtf(dot(v, v));
}
__device__ __forceinline__ float3 normalize(float3 v) {
    return v * rsqrtf(dot(v, v));
}

__device__ __forceinline__ float4 safe_normalize(float4 v) {
    float sq = dot(v, v);
    return (sq > 0.0f) ? v * rsqrtf(sq) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}
__device__ __forceinline__ float3 safe_normalize(float3 v) {
    float sq = dot(v, v);
    return (sq > 0.0f) ? v * rsqrtf(sq) : make_float3(0.0f, 0.0f, 0.0f);
}

__device__ __forceinline__ float4 lerp(float4 a, float4 b, float t) {
    return a + (b - a) * t;
}
__device__ __forceinline__ float3 lerp(float3 a, float3 b, float t) {
    return a + (b - a) * t;
}

__device__ __forceinline__ uint pack_unorm4x8(float4 v) {
    v = saturate(v) * 255.0f;
    return (uint(floorf(v.x + 0.5f)) << 0) | (uint(floorf(v.y + 0.5f)) << 8) |
           (uint(floorf(v.z + 0.5f)) << 16) | (uint(floorf(v.w + 0.5f)) << 24);
}

__device__ __forceinline__ float4 unpack_unorm4x8(uint packed) {
    return make_float4(((packed >> 0) & 0xFF) * (1.0f / 255.0f),
                       ((packed >> 8) & 0xFF) * (1.0f / 255.0f),
                       ((packed >> 16) & 0xFF) * (1.0f / 255.0f),
                       ((packed >> 24) & 0xFF) * (1.0f / 255.0f));
}

struct mat4 {
    float4 c0, c1, c2, c3;  // columns

    __device__ __forceinline__ float4 operator*(float4 v) const {
        return c0 * v.x + c1 * v.y + c2 * v.z + c3 * v.w;
    }
};

__device__ __forceinline__ mat4 look_at(vec3 eye, vec3 center, vec3 up) {
    vec3 f = safe_normalize(center - eye);
    vec3 s = safe_normalize(cross(f, up));
    vec3 u = cross(s, f);

    mat4 m;
    m.c0 = make_float4(s.x, u.x, -f.x, 0.0f);
    m.c1 = make_float4(s.y, u.y, -f.y, 0.0f);
    m.c2 = make_float4(s.z, u.z, -f.z, 0.0f);
    m.c3 = make_float4(-dot(s, eye), -dot(u, eye), dot(f, eye), 1.0f);
    return m;
}
}  // namespace cuda_math