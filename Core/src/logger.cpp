#include "logger.h"

#include "spdlog/sinks/stdout_color_sinks.h"

std::shared_ptr<spdlog::logger> logger::s_core_logger;

void logger::init() {
    spdlog::set_pattern("%^[%T] %n: %v%$");
    s_core_logger = spdlog::stdout_color_mt("CORE");
    s_core_logger->set_level(spdlog::level::trace);
}
