#include "kamanri/window_procedures/wingdi_window/update_procedure.hpp"
#include "kamanri/utils/string.hpp"
#include "kamanri/utils/log.hpp"
#include "kamanri/utils/result.hpp"

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


		}
	}
}

using namespace Kamanri::Renderer::World;
using namespace Kamanri::Windows::WinGDI_Window$;
using namespace Kamanri::WindowProcedures::WinGDI_Window;
using namespace Kamanri::Utils;

UpdateProcedure::UpdateProcedure(int(*update_func)(Kamanri::Renderer::World::World3D&), unsigned int screen_width, unsigned int screen_height, bool is_offline, unsigned int frame_count, unsigned int wait_millis)
{
	_update_func = update_func;
	_screen_width = screen_width;
	_screen_height = screen_height;
	_is_offline = is_offline;

	if(is_offline && frame_count < 1) 
		Log::Error(__UpdateProcedure::LOG_NAME, "Invalid frame_count: %u", frame_count);
	
	_frame_count = frame_count;
	_wait_millis = wait_millis;
	_frames = NewArray<P<unsigned long[]>>(_frame_count);
	for(size_t i = 0; i < _frame_count; i++)
	{
		_frames[i] = NewArray<unsigned long>(_screen_width * _screen_height);
	}
}

UpdateProcedure::UpdateProcedure(UpdateProcedure const& other)
{
	_update_func = other._update_func;
	_screen_width = other._screen_width;
	_screen_height = other._screen_height;
}


void UpdateProcedure::Func(Kamanri::Windows::WinGDI_Window$::WinGDI_Message& message)
{
	InvokeNext(message);
	if (message.u_msg == WM_CLOSE)
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
		int update_res;
		//

		if (!_is_offline)
		{
			while (this->_is_window_alive)
			{
				// this part rightly belongs to DrawFunc
				update_res = _update_func(*message.world);
				if (update_res != 0)
				{
					Log::Error(__UpdateProcedure::LOG_NAME, "Failed to execute the update_func caused by:");
					exit(update_res);
				}
				//
				Log::Debug(__UpdateProcedure::LOG_NAME, "Start to render...");

				painter.DrawFrom(message.world->Bitmap());

				painter.Flush();
				painter_factor.Clean(painter);
				Log::Debug(__UpdateProcedure::LOG_NAME, "Finish a frame render.");
			}
		}
		else
		{
			// offline rendering.
			for(unsigned int i = 0; _is_window_alive && i < _frame_count; i++)
			{
				update_res = _update_func(*message.world);
				if (update_res != 0)
				{
					Log::Error(__UpdateProcedure::LOG_NAME, "Failed to execute the update_func caused by:");
					exit(update_res);
				}
				// move result to frame.
				message.world->Bitmap()[0] = 0xffffff;
				memcpy(_frames[i].get(), message.world->Bitmap(), _screen_width * _screen_height * sizeof(unsigned long)); 

				Log::Debug(__UpdateProcedure::LOG_NAME, "Render process: %u / %u", i + 1, _frame_count);
			}

			Log::Debug(__UpdateProcedure::LOG_NAME, "Finished offline render, begin to draw...");

			unsigned int current_frame = 0;

			while (this->_is_window_alive)
			{

				painter.DrawFrom(_frames[current_frame].get());

				painter.Flush();

				Sleep(_wait_millis);

				painter_factor.Clean(painter);
				
				(++current_frame) %= _frame_count;
			}
		}
		
	});

}