#pragma once

#ifndef SWIG

#include "kamanri/windows/wingdi_window$.hpp"

#endif

namespace Kamanri
{
	namespace Windows
	{

		/// @brief This WinGDI_Window class is a adapter of WinGDI interfaces.
		class WinGDI_Window
		{
		public:
			WinGDI_Window(HINSTANCE h_instance, Kamanri::Renderer::World::World3D& world, unsigned int window_width = 600, unsigned int window_height = 600);
			~WinGDI_Window();
			WinGDI_Window& AddProcedure(Kamanri::Utils::Delegate<Kamanri::Windows::WinGDI_Window$::WinGDI_Message>::ANode&& proc);
			WinGDI_Window& Show();
			WinGDI_Window& Update();

			static void MessageLoop();

		private:
			// The world
			Kamanri::Renderer::World::World3D& _world;

			friend LRESULT __stdcall Kamanri::Windows::WinGDI_Window$::WindowProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
			// handle of window
			HWND _h_wnd;
			// window message process callback
			LRESULT(*_WindowProc)
			(HWND, UINT, WPARAM, LPARAM);
			// delegate chain
			Kamanri::Utils::Delegate<WinGDI_Window$::WinGDI_Message> _procedure_chain;

			unsigned int _window_width;
			unsigned int _window_height;

		};

	}

}
