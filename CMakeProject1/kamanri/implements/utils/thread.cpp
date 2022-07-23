#include "../../utils/thread.h"
#include <thread>
#include <chrono>

void Kamanri::Utils::Thread::Sleep(int millis) {
    std::this_thread::sleep_for(std::chrono::milliseconds(millis));
}