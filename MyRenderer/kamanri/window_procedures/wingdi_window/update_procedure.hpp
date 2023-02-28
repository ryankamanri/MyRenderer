#pragma once
#include <thread>
#include "kamanri/renderer/world/frame_buffer.hpp"
#include "kamanri/utils/logs.hpp"
#include "kamanri/utils/delegate.hpp"
#include "kamanri/windows/wingdi_window.hpp"

namespace Kamanri
{
	namespace WindowProcedures
	{
		namespace WinGDI_Window
		{
			namespace __UpdateProcedure
			{
				constexpr const char* LOG_NAME = STR(Kamanri::WindowProcedures::WinGDI_Window::UpdateProcedure);
			} // namespace __UpdateProcedure

			using namespace Renderer::World;
			using namespace Kamanri::Windows::WinGDI_Window$;
			using namespace Utils;

			class UpdateProcedure: public Kamanri::Utils::Delegate<Kamanri::Windows::WinGDI_Window$::WinGDI_Message>::ANode
			{
				public:
				UpdateProcedure(DefaultResult(*update_func)(World3D&), unsigned int screen_width, unsigned int screen_height)
				{
					_update_func = update_func;
					_screen_width = screen_width;
					_screen_height = screen_height;
				}


				private:
				DefaultResult(*_update_func)(World3D&) = nullptr;
				std::thread _update_thread;
				unsigned int _screen_width;
				unsigned int _screen_height;
				bool _is_window_alive = true;
				bool _is_thread_running = false;

				void Func(Kamanri::Windows::WinGDI_Window$::WinGDI_Message& message)
				{
					InvokeNext(message);
					if(message.u_msg == WM_CLOSE)
					{
						_is_window_alive = false;
						_update_thread.join();
						Log::Info(__UpdateProcedure::LOG_NAME, "Update thread exited");
						return;
					}

					if (message.u_msg != WM_PAINT || _is_thread_running) return;

					_is_thread_running = true;

					_update_thread = std::thread([this, message]()
					{
						// while vars
						PainterFactor painter_factor(message.h_wnd, _screen_width, _screen_height);
						auto painter = painter_factor.CreatePainter();

						// int color;
						DefaultResult update_res;
						//
						while (this->_is_window_alive)
						{
							// this part rightly belongs to DrawFunc
							update_res = _update_func(message.world);
							if (update_res.IsException())
							{
								Log::Error(__UpdateProcedure::LOG_NAME, "Failed to execute the update_func caused by:");
								update_res.Print();
								exit(update_res.Code());
							}
							//
							Log::Debug(__UpdateProcedure::LOG_NAME, "Start to render...");

							painter.DrawFrom(message.world.Bitmap());

							painter.Flush();
							painter_factor.Clean(painter);
							Log::Debug(__UpdateProcedure::LOG_NAME, "Finish a frame render.");
						}
					});

				}

			};
		} // namespace WinGDI_Window

	} // namespace WindowProcedures

} // namespace Kamanri
