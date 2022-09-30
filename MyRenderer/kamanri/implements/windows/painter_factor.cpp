#include "kamanri/windows/painter_factor.hpp"

using namespace Kamanri::Windows;

namespace Kamanri
{   
    namespace Windows
    {
        namespace __PainterFactor
        {
            int WINDOW_WIDTH;
            int WINDOW_HEIGHT;

        } // namespace __PainterFactor
        
    } // namespace Windows
    
} // namespace Kamanri



PainterFactor::PainterFactor(HWND h_wnd, int window_width, int window_height): _h_wnd(h_wnd)
{
    __PainterFactor::WINDOW_WIDTH = window_width;
    __PainterFactor::WINDOW_HEIGHT = window_height;

    _h_dc = GetDC(_h_wnd);

    _h_black_dc = CreateCompatibleDC(_h_dc);
    HBITMAP hBackBmp = CreateCompatibleBitmap(_h_dc, __PainterFactor::WINDOW_WIDTH, __PainterFactor::WINDOW_HEIGHT);
    SelectObject(_h_black_dc, hBackBmp);
}

PainterFactor::~PainterFactor()
{
    ReleaseDC(_h_wnd, _h_dc);
}

void PainterFactor::Clean(Painter& painter)
{
    BitBlt(painter._h_draw_dc, 0, 0, __PainterFactor::WINDOW_WIDTH, __PainterFactor::WINDOW_HEIGHT, _h_black_dc, 0, 0, SRCCOPY);
}

Painter PainterFactor::CreatePainter()
{
    // on paint
    //创建内存DC（先放到内存中）
    HDC h_draw_dc = CreateCompatibleDC(_h_dc);

    //创建一张兼容位图
    // note:
    //这要注意,如果创建和内存DC兼容的位图就只有黑白色,不会有彩色
    //所以要创建实际对象DC.窗口DC或静态控件DC兼容的内存位图
    HBITMAP hBackBmp = CreateCompatibleBitmap(_h_dc, __PainterFactor::WINDOW_WIDTH, __PainterFactor::WINDOW_HEIGHT);

    SelectObject(h_draw_dc, hBackBmp);

    return Painter(_h_dc, h_draw_dc, __PainterFactor::WINDOW_WIDTH, __PainterFactor::WINDOW_HEIGHT);
}

