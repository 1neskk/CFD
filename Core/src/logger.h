#pragma once

#include <memory>
#include <chrono>

#include "spdlog/spdlog.h"
#include "spdlog/fmt/ostr.h"

class logger {
public:
    static void init();

    static std::shared_ptr<spdlog::logger>& get_core_logger() { return s_core_logger; }

private:
    static std::shared_ptr<spdlog::logger> s_core_logger;
};

// Core log macros
#define LOG_TRACE(...)    ::logger::get_core_logger()->trace(__VA_ARGS__)
#define LOG_INFO(...)     ::logger::get_core_logger()->info(__VA_ARGS__)
#define LOG_WARN(...)     ::logger::get_core_logger()->warn(__VA_ARGS__)
#define LOG_ERROR(...)    ::logger::get_core_logger()->error(__VA_ARGS__)
#define LOG_CRITICAL(...) ::logger::get_core_logger()->critical(__VA_ARGS__)

// Throttled log macros
#define LOG_INFO_THROTTLED(delay_ms, ...) \
    do { \
        static auto last_log_time = std::chrono::steady_clock::now(); \
        auto current_time = std::chrono::steady_clock::now(); \
        if (std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_log_time).count() > delay_ms) { \
            LOG_INFO(__VA_ARGS__); \
            last_log_time = current_time; \
        } \
    } while(0)
