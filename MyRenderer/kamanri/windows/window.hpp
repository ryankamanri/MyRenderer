#pragma once
#include <windows.h>
#include <thread>
#include "painter.hpp"



namespace Kamanri
{
    namespace Windows
    {

        class Window
        {
        public:
            Window(HINSTANCE h_instance, int window_width = 600, int window_height = 600);
            ~Window();
            bool Show();
            bool Update();
            void (*DrawFunc)(PainterFactor painter_factor);

            static void MessageLoop();
            // callback paint function called by WindowProc, do not call it outside
            void _Paint();

        private:
            // handle of window
            HWND _h_wnd;
            // window message process callback
            LRESULT CALLBACK (*_WindowProc)(HWND, UINT, WPARAM, LPARAM);

            std::thread _paint_thread;
        };

    }

}
