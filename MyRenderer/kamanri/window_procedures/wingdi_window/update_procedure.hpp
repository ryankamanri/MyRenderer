#pragma once
#ifndef SWIG
#include <thread>
#include "kamanri/renderer/world/frame_buffer.hpp"
#include "kamanri/utils/delegate.hpp"
#include "kamanri/windows/wingdi_window.hpp"
#endif

namespace Kamanri
{
	namespace WindowProcedures
	{
		namespace WinGDI_Window
		{

			class UpdateProcedure: public Kamanri::Utils::Delegate<Kamanri::Windows::WinGDI_Window$::WinGDI_Message>::ANode
			{
				public:
				UpdateProcedure(Kamanri::Utils::DefaultResult (*update_func)(Kamanri::Renderer::World::World3D&), unsigned int screen_width, unsigned int screen_height);
				UpdateProcedure(UpdateProcedure const& other);

				private:
				Kamanri::Utils::DefaultResult (*_update_func)(Kamanri::Renderer::World::World3D&) = nullptr;
				std::thread _update_thread;
				unsigned int _screen_width;
				unsigned int _screen_height;
				bool _is_window_alive = true;
				bool _is_thread_running = false;

				void Func(Kamanri::Windows::WinGDI_Window$::WinGDI_Message& message);


			};
		} // namespace WinGDI_Window

	} // namespace WindowProcedures

} // namespace Kamanri
