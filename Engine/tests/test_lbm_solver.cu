#include <gtest/gtest.h>
#include "Physics/lbm_solver.cuh"
#include <vector>
#include <cmath>
#include <numeric>

TEST(LBMSolverTest, Initialization) {
    int width = 128;
    int height = 128;
    lbm_solver solver(width, height);
    
    EXPECT_EQ(solver.get_width(), width);
    EXPECT_EQ(solver.get_height(), height);
    
    // Initialize
    solver.init();
    
    // Check if density buffer is accessible and has correct size
    // Note: get_density() returns a reference to cuda_buffer
    // We can't check internal size of cuda_buffer easily without a getter, 
    // but we can try to copy from it.
    
    std::vector<float> h_rho(width * height);
    // This copy should succeed if allocated
    solver.get_density().copy_to_host(h_rho.data(), width * height);
}

TEST(LBMSolverTest, ResetAndStep) {
    int width = 64;
    int height = 64;
    lbm_solver solver(width, height);
    solver.init();
    solver.reset();
    
    std::vector<float> h_rho_initial(width * height);
    solver.get_density().copy_to_host(h_rho_initial.data(), width * height);
    
    // Calculate initial mass
    double initial_mass = std::accumulate(h_rho_initial.begin(), h_rho_initial.end(), 0.0);
    
    // Run steps
    for (int i = 0; i < 10; ++i) {
        solver.step();
    }
    
    std::vector<float> h_rho_final(width * height);
    solver.get_density().copy_to_host(h_rho_final.data(), width * height);
    
    double final_mass = std::accumulate(h_rho_final.begin(), h_rho_final.end(), 0.0);
    
    // Check for NaNs
    EXPECT_FALSE(std::isnan(final_mass));
    
    // Mass conservation check (approximate, LBM should conserve mass)
    // Allow small error due to float precision
    EXPECT_NEAR(initial_mass, final_mass, 1.0); 
}
