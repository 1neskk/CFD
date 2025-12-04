#include <gtest/gtest.h>
// #include "Graphics/vulkan_cuda_buffer.cuh" // Uncomment when ready to test

TEST(InteropTest, BufferCreation) {
    // Similar to Renderer, Interop requires Vulkan Device.
    GTEST_SKIP() << "Interop tests require a running Application/Vulkan context. Skipping for now.";
}
