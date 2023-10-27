#pragma once
#ifndef SWIG
#include <thread>
#include "kamanri/renderer/world/frame_buffer.hpp"
#include "kamanri/utils/delegate.hpp"
#include "kamanri/windows/wingdi_window.hpp"
#include "kamanri/utils/memory.hpp"
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
				UpdateProcedure(int (*update_func)(Kamanri::Renderer::World::World3D&), unsigned int screen_width, unsigned int screen_height, bool is_offline = false, unsigned int frame_count = 1, unsigned int wait_millis = 10);
				UpdateProcedure(UpdateProcedure const& other);

				private:
				int (*_update_func)(Kamanri::Renderer::World::World3D&) = nullptr;
				std::thread _update_thread;
				unsigned int _screen_width;
				unsigned int _screen_height;
				bool _is_window_alive = true;
				bool _is_thread_running = false;
				// offline rendering
				bool _is_offline = false;
				unsigned int _frame_count = 1;
				unsigned int _wait_millis = 10;
				Kamanri::Utils::P<Kamanri::Utils::P<unsigned long[]>[]> _frames;


				void Func(Kamanri::Windows::WinGDI_Window$::WinGDI_Message& message);


			};
		} // namespace WinGDI_Window

	} // namespace WindowProcedures

} // namespace Kamanri
