#pragma once
#include <windows.h>
#include <thread>
#include "painter_factor.hpp"

namespace Kamanri
{
    namespace Windows
    {

        class Painter
        {
        public:
            Painter(HDC h_dc, HDC h_mem_dc, int window_width, int window_height);
            ~Painter();
            WINBOOL Flush() const;
            inline COLORREF Dot(int x, int y, COLORREF color) const { return SetPixel(_h_draw_dc, x, y, color); };
            friend void PainterFactor::Clean(Painter &painter);

        private:
            HDC _h_dc;
            HDC _h_draw_dc;
        };
    }
}