#pragma once
#include <windows.h>
#include <thread>

namespace Kamanri
{
    namespace Windows
    {
        namespace WinGDI_Window$
        {
            // TODO: let Painter, PainterFactor and Window extend from object in 'window_procedures/window.hpp'
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

            class Painter
            {
            public:
                Painter(HDC h_dc, HDC h_mem_dc, int window_width, int window_height);
                ~Painter();
                BOOL Flush() const;
                inline COLORREF Dot(int x, int y, COLORREF color) const { return SetPixel(_h_draw_dc, x, y, color); };
                friend void PainterFactor::Clean(Painter &painter);

            private:
                HDC _h_dc;
                HDC _h_draw_dc;
            };
        } // namespace WinGDI_Window$

        class WinGDI_Window
        {
        public:
            WinGDI_Window(HINSTANCE h_instance, int window_width = 600, int window_height = 600);
            ~WinGDI_Window();
            bool Show();
            bool Update();
            void (*DrawFunc)(WinGDI_Window$::PainterFactor painter_factor);

            static void MessageLoop();
            // callback paint function called by WindowProc, do not call it outside
            void _Paint();

        private:
            // handle of window
            HWND _h_wnd;
            // window message process callback
            LRESULT(*_WindowProc)
            (HWND, UINT, WPARAM, LPARAM);

            std::thread _paint_thread;
        };

    }

}
