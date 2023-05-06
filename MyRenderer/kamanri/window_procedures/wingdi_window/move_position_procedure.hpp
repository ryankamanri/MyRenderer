#pragma once
#ifndef SWIG

#include <thread>
#include "kamanri/renderer/world/frame_buffer.hpp"
#include "kamanri/utils/log.hpp"
#include "kamanri/utils/delegate.hpp"
#include "kamanri/windows/wingdi_window.hpp"

#endif

namespace Kamanri
{
	namespace WindowProcedures
	{
		namespace WinGDI_Window
		{

			class MovePositionProcedure: public Kamanri::Utils::Delegate<Kamanri::Windows::WinGDI_Window$::WinGDI_Message>::ANode
			{
				public:
				MovePositionProcedure(double min_move_location, double min_move_theta);
			

				private:
				double _min_move_location;
				double _min_move_direction;
				Kamanri::Maths::Vector _next_direction;
				Kamanri::Maths::SMatrix _turn_left, _turn_right, _turn_up, _turn_down;
				Kamanri::Maths::SMatrix _move;


				void Func(Kamanri::Windows::WinGDI_Window$::WinGDI_Message& message);
	
			};
		} // namespace WinGDI_Window

	} // namespace WindowProcedures

} // namespace Kamanri
