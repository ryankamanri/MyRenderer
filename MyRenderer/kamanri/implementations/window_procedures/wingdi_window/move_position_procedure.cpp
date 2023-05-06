#include "kamanri/window_procedures/wingdi_window/move_position_procedure.hpp"
#include "kamanri/utils/string.hpp"

namespace Kamanri
{
	namespace WindowProcedures
	{
		namespace WinGDI_Window
		{
			namespace __MovePositionProcedure
			{
				constexpr const char* LOG_NAME = STR(Kamanri::WindowProcedures::WinGDI_Window::MovePositionProcedure);

				constexpr const WPARAM W_KEY = 0x57;
				constexpr const WPARAM A_KEY = 0x41;
				constexpr const WPARAM S_KEY = 0x53;
				constexpr const WPARAM D_KEY = 0x44;
				constexpr const WPARAM Q_KEY = 0x51;
				constexpr const WPARAM E_KEY = 0x45;
				constexpr const WPARAM UP_KEY = 0x26;
				constexpr const WPARAM DOWN_KEY = 0x28;
				constexpr const WPARAM LEFT_KEY = 0x25;
				constexpr const WPARAM RIGHT_KEY = 0x27;

				constexpr const int X = 0;
				constexpr const int Y = 1;
				constexpr const int Z = 2;
			} // namespace __MovePositionProcedure
		} // namespace WinGDI_Window

	} // namespace WindowProcedures

} // namespace Kamanri

using namespace Kamanri::Renderer::World;
using namespace Kamanri::Windows::WinGDI_Window$;
using namespace Kamanri::WindowProcedures::WinGDI_Window;
using namespace Kamanri::Utils;


MovePositionProcedure::MovePositionProcedure(double min_move_location, double min_move_theta)
{
	if (min_move_theta <= 0 || min_move_theta <= 0)
	{
		Log::Error(__MovePositionProcedure::LOG_NAME, "Invalid min_move_direction <= 0 || min_move_direction <= 0");
	}
	_min_move_location = min_move_location;
	_min_move_direction = min_move_theta;

	_turn_left =
	{
		cos(min_move_theta), 0, sin(min_move_theta), 0,
		0, 1, 0, 0,
		-sin(min_move_theta), 0, cos(min_move_theta), 0,
		0, 0, 0, 1
	};

	_turn_right =
	{
		cos(min_move_theta), 0, -sin(min_move_theta), 0,
		0, 1, 0, 0,
		sin(min_move_theta), 0, cos(min_move_theta), 0,
		0, 0, 0, 1
	};

	_turn_up =
	{
		1, 0, 0, 0,
		0, cos(min_move_theta), -sin(min_move_theta), 0,
		0, sin(min_move_theta), cos(min_move_theta), 0,
		0, 0, 0, 1
	};

	_turn_down =
	{
		1, 0, 0, 0,
		0, cos(min_move_theta), sin(min_move_theta), 0,
		0, -sin(min_move_theta), cos(min_move_theta), 0,
		0, 0, 0, 1
	};

}



void MovePositionProcedure::Func(Kamanri::Windows::WinGDI_Window$::WinGDI_Message& message)
{
	using namespace __MovePositionProcedure;
	if (message.u_msg == WM_KEYDOWN)
	{

		auto& location = message.world->GetCamera().Location();
		auto& direction = message.world->GetCamera().Direction();
		auto& upward = message.world->GetCamera().Upward();
		_next_direction = direction;

		switch (message.w_param)
		{
			case W_KEY:
				Log::Info(__MovePositionProcedure::LOG_NAME, "Move forward");
				_move =
				{
					1, 0, 0, direction[X] * _min_move_location,
					0, 1, 0, direction[Y] * _min_move_location,
					0, 0, 1, direction[Z] * _min_move_location,
					0, 0, 0, 1
				};
				_move* location;
				break;
			case A_KEY:
				Log::Info(__MovePositionProcedure::LOG_NAME, "Move left");
				_next_direction = upward;
				_next_direction *= direction;
				_next_direction.Unitization();
				_move =
				{
					1, 0, 0, _next_direction[X] * _min_move_location,
					0, 1, 0, _next_direction[Y] * _min_move_location,
					0, 0, 1, _next_direction[Z] * _min_move_location,
					0, 0, 0, 1
				};
				_move* location;
				break;
			case S_KEY:
				Log::Info(__MovePositionProcedure::LOG_NAME, "Move backward");
				_move =
				{
					1, 0, 0, -direction[X] * _min_move_location,
					0, 1, 0, -direction[Y] * _min_move_location,
					0, 0, 1, -direction[Z] * _min_move_location,
					0, 0, 0, 1
				};
				_move* location;
				break;
			case D_KEY:
				Log::Info(__MovePositionProcedure::LOG_NAME, "Move right");
				_next_direction = direction;
				_next_direction *= upward;
				_next_direction.Unitization();
				_move =
				{
					1, 0, 0, _next_direction[X] * _min_move_location,
					0, 1, 0, _next_direction[Y] * _min_move_location,
					0, 0, 1, _next_direction[Z] * _min_move_location,
					0, 0, 0, 1
				};
				_move* location;
				break;

			case Q_KEY:
				Log::Info(__MovePositionProcedure::LOG_NAME, "Move up");
				_move =
				{
					1, 0, 0, upward[X] * _min_move_location,
					0, 1, 0, upward[Y] * _min_move_location,
					0, 0, 1, upward[Z] * _min_move_location,
					0, 0, 0, 1
				};
				_move* location;
				break;
			case E_KEY:
				Log::Info(__MovePositionProcedure::LOG_NAME, "Move down");
				_move =
				{
					1, 0, 0, -upward[X] * _min_move_location,
					0, 1, 0, -upward[Y] * _min_move_location,
					0, 0, 1, -upward[Z] * _min_move_location,
					0, 0, 0, 1
				};
				_move* location;
				break;
			case UP_KEY:
				Log::Info(__MovePositionProcedure::LOG_NAME, "Turn up");
				_turn_up* direction;
				break;
			case DOWN_KEY:
				Log::Info(__MovePositionProcedure::LOG_NAME, "Turn down");
				_turn_down* direction;
				break;
			case LEFT_KEY:
				Log::Info(__MovePositionProcedure::LOG_NAME, "Turn left");
				_turn_left* direction;
				break;
			case RIGHT_KEY:
				Log::Info(__MovePositionProcedure::LOG_NAME, "Turn right");
				_turn_right* direction;
				break;

			default:
				break;
		}

	}
	InvokeNext(message);
}

