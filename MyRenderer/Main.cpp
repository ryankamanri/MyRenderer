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

constexpr const char* OBJ_PATH = "../../out/floor.obj";
constexpr const char* TGA_PATH = "../../out/floor_diffuse.tga";




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

	camera.Transform();

	world.Build();

	return DEFAULT_RESULT;
}

void StartRender(HINSTANCE hInstance)
{
	WinGDI_Window(hInstance, WINDOW_LENGTH, WINDOW_LENGTH)
		.SetWorld(
			World3D(
				Camera(
					{ 0, 0, 4, 1 },
					{ 0, 0, -1, 0 },
					{ 0, 1, 0, 0 },
					-1, 
					-10,
					WINDOW_LENGTH,
					WINDOW_LENGTH
				)
			).AddObjModel(
				ObjModel("../../out/diablo3_pose.obj", "../../out/diablo3_pose_diffuse.tga"),
				{
					2, 0, 0, 0,
				 	0, 2, 0, 0,
				 	0, 0, 2, 0,
				 	0, 0, 0, 1 
				}
			)
			// .AddObjModel(
			// 	ObjModel("../../out/skybox.obj", "../../out/skybox.tga"),
			// 	{
			// 		10, 0, 0, 0,
			// 	 	0, 10, 0, 0,
			// 	 	0, 0, 10, 0,
			// 	 	0, 0, 0, 1 
			// 	}
			// )
			).AddProcedure(
				UpdateProcedure(UpdateFunc, WINDOW_LENGTH, WINDOW_LENGTH)
			).Show().MessageLoop();
}

//////////////////////////////////////////////////////

int WinMain(HINSTANCE hInstance, HINSTANCE hPreInstance, LPSTR lpCmdLine, int nCmdShow)
{
	// set the log level and is_print
	Log::SetLevel(Log$::DEBUG_LEVEL);

	StartRender(hInstance);

	system("pause");
	return 0;
}
