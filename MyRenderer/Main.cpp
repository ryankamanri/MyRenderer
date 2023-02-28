#include <iostream>
#include <thread>
#include <cfloat>
#include <cmath>
#include "kamanri/all.hpp"
#include "cuda_dll/foo.hpp"
#include "cuda_dll/exports/set_log_level.hpp"

using namespace Kamanri::Utils;
using namespace Kamanri::Maths;
using namespace Kamanri::Windows;
using namespace Kamanri::Windows::WinGDI_Window$;
using namespace Kamanri::WindowProcedures::WinGDI_Window;
using namespace Kamanri::Renderer;
using namespace Kamanri::Renderer::World;


constexpr const char* LOG_NAME = "Main";
const int WINDOW_LENGTH = 400;

constexpr const char* OBJ_PATH = "../../out/diablo3_pose.obj";
constexpr const char* TGA_PATH = "../../out/diablo3_pose_diffuse.tga";
constexpr const char* OBJ2_PATH = "../../out/floor.obj";
constexpr const char* TGA2_PATH = "../../out/floor_diffuse.tga";




namespace __UpdateFunc
{
	Vector direction(4);
	double theta = PI / 64;
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
	World3D world(
		Camera(
			{ 0, 0, 4, 1 },
			{ 0, 0, -1, 0 },
			{ 0, 1, 0, 0 },
			-1,
			-10,
			WINDOW_LENGTH,
			WINDOW_LENGTH
		),
		BlingPhongReflectionModel({
			BlingPhongReflectionModel$::PointLight({0, 3, 4, 1}, 400, 0xffffff)
		}, WINDOW_LENGTH, WINDOW_LENGTH, 0.95, 1 / PI * 2, 0.1)
	);
	world.AddObjModel(
		ObjModel(OBJ_PATH, TGA_PATH),
		{
			2, 0, 0, 0,
			0, 2, 0, 0,
			0, 0, 2, 0,
			0, 0, 0, 1
		}
	).AddObjModel(
		ObjModel(OBJ2_PATH, TGA2_PATH),
		{
			2, 0, 0, 0,
			0, 2, 0, 0,
			0, 0, 2, 0,
			0, 0, 0, 1
		}
	).Commit();
	WinGDI_Window(hInstance, world, WINDOW_LENGTH, WINDOW_LENGTH)
		.AddProcedure(
			UpdateProcedure(UpdateFunc, WINDOW_LENGTH, WINDOW_LENGTH)
		).Show().MessageLoop();
}



void SetLevel(LogLevel level)
{
	Log::SetLevel(level);
	dll cuda_dll;
	func_type(SetLogLevel) set_log_level;
	load_dll(cuda_dll, cuda_dll, LOG_NAME);
	import_func(SetLogLevel, cuda_dll, set_log_level, LOG_NAME);
	set_log_level(level);
}



//////////////////////////////////////////////////////

int main()

{
	HINSTANCE instance = GetModuleHandle("MyRenderer.exe");
	// set the log level and is_print
	SetLevel(Log$::DEBUG_LEVEL);
	Log::Info(LOG_NAME, "%p: May you have a nice day!", instance);

	StartRender(instance);

	return 0;
}
