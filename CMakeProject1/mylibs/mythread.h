#pragma once
#include <thread>
#include <chrono>

void Sleep(int millis) {
    std::this_thread::sleep_for(std::chrono::milliseconds(millis));
}