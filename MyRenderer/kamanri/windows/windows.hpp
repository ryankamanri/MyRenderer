#pragma once
#include <windows.h>
#include <thread>

namespace Kamanri
{
    namespace Windows
    {
        namespace Windows
        {
            class Painter
            {
            public:
                Painter(HDC h_dc, HDC h_mem_dc);
                WINBOOL Flush() const;
                inline COLORREF Dot(int x, int y, COLORREF color) const { return SetPixel(_h_mem_dc, x, y, color); };

            private:
                HDC _h_dc;
                HDC _h_mem_dc;
            };

            class Window
            {
            public:
                Window(HINSTANCE h_instance);
                ~Window();
                bool Show();
                bool Update();
                void (*DrawFunc)(Painter painter);

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
