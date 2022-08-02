#pragma once
#include <windows.h>
#include <thread>

namespace Kamanri
{
    namespace Windows
    {
        namespace Windows
        {
            class Painter;
            class PainterFactor
            {
                public:
                    PainterFactor(HWND h_wnd);
                    ~PainterFactor();
                    Painter CreatePainter();
                    void Clean(Painter& painter);
                private:
                    HWND _h_wnd;
                    HDC _h_dc;
                    HDC _h_black_dc;
            };
            
            class Painter
            {
            public:
                Painter(HDC h_dc, HDC h_mem_dc);
                ~Painter();
                WINBOOL Flush() const;
                inline COLORREF Dot(int x, int y, COLORREF color) const { return SetPixel(_h_draw_dc, x, y, color); };
                friend void PainterFactor::Clean(Painter& painter);

            private:
                HDC _h_dc;
                HDC _h_draw_dc;
            };

            

            class Window
            {
            public:
                Window(HINSTANCE h_instance);
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

}
