#include <iostream>
#include <thread>
#include <cfloat>
#include <cmath>
#include "kamanri/all.hpp"

using namespace Kamanri::Utils;
using namespace Kamanri::Maths;
using namespace Kamanri::Windows;
using namespace Kamanri::Windows::WinGDI_Window$;
using namespace Kamanri::WindowProcedures::WinGDI_Window;
using namespace Kamanri::Renderer;
using namespace Kamanri::Renderer::World;


constexpr const char* LOG_NAME = "Main";
const int WINDOW_LENGTH = 600;

constexpr const char* OBJ_PATH = "../../out/african_head.obj";
constexpr const char* TGA_PATH = "../../out/african_head_diffuse.tga";

bool is_print = false;


namespace __UpdateFunc
{
	Vector direction(4);
	double theta = PI / 16;
	SMatrix revolve_matrix =
	{
		cos(theta), 0, -sin(theta), 0,
		0, 1, 0, 0,
		sin(theta), 0, cos(theta), 0,
		0, 0, 0, 1
	};
	// SMatrix revolve_matrix =
	// {
	// 	1, 0, 0, 0,
	// 	0, cos(theta), -sin(theta), 0,
	// 	0, sin(theta), cos(theta), 0,
	// 	0, 0, 0, 1
	// };
} // namespace __UpdateFunc

DefaultResult UpdateFunc(World3D& world)
{
	using namespace __UpdateFunc;
	Camera& camera = world.GetCamera();
	direction = camera.Direction();

	revolve_matrix* camera.Direction();
	revolve_matrix* camera.Location();

	camera.InverseUpperByDirection(direction);

	camera.Transform(is_print);

	world.Build(is_print);

	return DEFAULT_RESULT;
}

void StartRender(HINSTANCE hInstance)
{
	WinGDI_Window(hInstance, WINDOW_LENGTH, WINDOW_LENGTH)
		.SetWorld(
			World3D(
				Camera(
					{ 0, 0, 3, 1 },
					{ 0, 0, -1, 0 },
					{ 0, 1, 0, 0 },
					-1, 
					-10,
					WINDOW_LENGTH,
					WINDOW_LENGTH
				)
			).AddObjModel(
				ObjModel(OBJ_PATH, TGA_PATH),
				{ 2, 0, 0, 0,
				 0, 2, 0, 0,
				 0, 0, 2, 0,
				 0, 0, 0, 1 }
			)).AddProcedure(
				UpdateProcedure(UpdateFunc, WINDOW_LENGTH, WINDOW_LENGTH)
			).Show().MessageLoop();
}

//////////////////////////////////////////////////////

int WinMain(HINSTANCE hInstance, HINSTANCE hPreInstance, LPSTR lpCmdLine, int nCmdShow)
{
	// set the log level and is_print
	Log::Level(Log$::DEBUG_LEVEL);
	is_print = false;

	StartRender(hInstance);

	system("pause");
	return 0;
}
