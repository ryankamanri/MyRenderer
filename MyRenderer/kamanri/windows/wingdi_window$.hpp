#pragma once

#ifndef SWIG

#include <Windows.h>
#include <thread>

#include "kamanri/utils/delegate.hpp"
#include "kamanri/renderer/world/world3d.hpp"

#endif

namespace Kamanri
{
	namespace Windows
	{
		namespace WinGDI_Window$
		{
			// TODO: let Painter, PainterFactor and Window extend from object in 'renderer/a_window.hpp'
			class Painter;
			class PainterFactor
			{
			public:
				PainterFactor(HWND h_wnd, unsigned int window_width, unsigned int window_height);
				~PainterFactor();
				Painter CreatePainter();
				void Clean(Painter &painter);

			private:
				HWND _h_wnd;
				HDC _h_dc;
				HDC _h_black_dc;
				unsigned int _window_width;
				unsigned int _window_height;
			};

			class Painter
			{
			public:
				Painter(HDC h_dc, HDC h_mem_dc, HBITMAP h_bitmap, unsigned int window_width, unsigned int window_height);
				~Painter();
				BOOL Flush() const;
				inline COLORREF Dot(int x, int y, COLORREF color) const { return SetPixel(_h_mem_dc, x, y, color); };
				int DrawFrom(DWORD* bitmap);
				friend void PainterFactor::Clean(Painter &painter);

			private:
				HDC _h_dc;
				HDC _h_mem_dc;
				HBITMAP _h_bitmap;
				BITMAPINFO _b_info;
				unsigned int _window_width;
				unsigned int _window_height;
			};

			class WinGDI_Message
			{
				public:
					/// @brief The reference of stored world.
					Kamanri::Renderer::World::World3D* world;

					// hWnd is a handle to the window.
					HWND h_wnd;
					// uMsg is the message code; for example, the WM_SIZE message indicates the window was resized.
					UINT u_msg;
					// wParam and lParam contain additional data that pertains to the message. The exact meaning depends on the message code.
					WPARAM w_param;
					LPARAM l_param;

					WinGDI_Message() = default;
					WinGDI_Message(Kamanri::Renderer::World::World3D& $world, HWND $h_wnd, UINT $u_msg, WPARAM $w_param, LPARAM $l_param): world(&$world), h_wnd($h_wnd), u_msg($u_msg), w_param($w_param), l_param($l_param) {}
			};

			LRESULT __stdcall WindowProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

		} // namespace WinGDI_Window$
	}
}