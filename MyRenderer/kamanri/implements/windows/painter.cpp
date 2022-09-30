#include "kamanri/windows/painter.hpp"

using namespace Kamanri::Windows;

namespace Kamanri
{   
    namespace Windows
    {
        namespace __Painter
        {
            int WINDOW_WIDTH;
            int WINDOW_HEIGHT;

        } // namespace __Painter
        
    } // namespace Windows
    
} // namespace Kamanri


Painter::Painter(HDC h_dc, HDC h_mem_dc, int window_width, int window_height) : _h_dc(h_dc), _h_draw_dc(h_mem_dc) 
{
    __Painter::WINDOW_WIDTH = window_width;
    __Painter::WINDOW_HEIGHT = window_height;
}

Painter::~Painter()
{
    DeleteObject(_h_draw_dc);
}

BOOL Painter::Flush() const { return BitBlt(_h_dc, 0, 0, __Painter::WINDOW_WIDTH, __Painter::WINDOW_HEIGHT, _h_draw_dc, 0, 0, SRCCOPY); }