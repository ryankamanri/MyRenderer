#pragma once
#include <windows.h>
#include <thread>


namespace Kamanri
{
    namespace Windows
    {
        class Painter;
        class PainterFactor
        {
        public:
            PainterFactor(HWND h_wnd, int window_width, int window_height);
            ~PainterFactor();
            Painter CreatePainter();
            void Clean(Painter &painter);

        private:
            HWND _h_wnd;
            HDC _h_dc;
            HDC _h_black_dc;
        };
    }
}

// DO NOT move this headfile to top cause it has the friend function of PaintFactor.
#include "painter.hpp"