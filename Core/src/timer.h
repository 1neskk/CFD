#pragma once

#include <chrono>
#include <iostream>
#include <string>

class timer {
   public:
    timer() { reset(); }

    void reset() {
        m_start_timepoint = std::chrono::high_resolution_clock::now();
    }

    float elapsed() const {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::high_resolution_clock::now() -
                   m_start_timepoint)
                   .count() *
               1e-9f;
    }

    float elapsed_ms() const { return elapsed() * 1000.0f; }

   private:
    std::chrono::time_point<std::chrono::high_resolution_clock>
        m_start_timepoint;
};

class timer_scope {
   public:
    timer_scope(const std::string &name) : m_name(name) {}

    ~timer_scope() {
        const float time = m_timer.elapsed_ms();
        std::cout << "[TIMER] " << m_name << ": " << time << "ms" << std::endl;
    }

   private:
    std::string m_name;
    timer m_timer;
};