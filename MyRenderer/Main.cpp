#include <iostream>
#include <thread>
#include <cfloat>
#include <cmath>
#include "kamanri/all.hpp"
using namespace Kamanri::Maths;
using namespace Kamanri::Windows;
using namespace Kamanri::Windows::WinGDI_Window$;
using namespace Kamanri::WindowProcedures::WinGDI_Window;
using namespace Kamanri::Renderer;
using namespace Kamanri::Renderer::World;


constexpr const char* LOG_NAME = "Main";
constexpr const int WINDOW_LENGTH = 400;
constexpr const bool IS_USE_CUDA = true;

#define BASE_PATH "C:/Users/97448/totFolder/source/repos/MyRenderer/MyRenderer/models/"

constexpr const char* OBJ_PATH = BASE_PATH "shiba/Shiba_Obj/Shiba.obj";
constexpr const char* TGA_PATH = BASE_PATH "shiba/Textures/Shiba_DIF01.tga";
constexpr const char* OBJ2_PATH = BASE_PATH "floor/floor.obj";
constexpr const char* TGA2_PATH = BASE_PATH "floor/floor_diffuse.tga";




namespace __UpdateFunc
{
	Vector direction(4);
	double theta = PI / 1024;
	SMatrix revolve_matrix =
	{
		cos(theta), 0, -sin(theta), 0,
		0, 1, 0, 0,
		sin(theta), 0, cos(theta), 0,
		0, 0, 0, 1
	};
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
			{ 0, -1, 5, 1 },
			{ 0, 0, -1, 0 },
			{ 0, 1, 0, 0 },
			-1,
			-5,
			WINDOW_LENGTH,
			WINDOW_LENGTH
		),
		BlingPhongReflectionModel({
			BlingPhongReflectionModel$::PointLight({0, 3, 4, 1}, 800, 0xffffff)
		}, WINDOW_LENGTH, WINDOW_LENGTH, 0.95, 1 / PI * 2, 0.2, IS_USE_CUDA),
		IS_USE_CUDA
	);
	world
	.AddObjModel(
		ObjModel(OBJ_PATH, TGA_PATH),
		{
			6, 0, 0, 0,
			0, 6, 0, -1.5,
			0, 0, 6, 1,
			0, 0, 0, 1
		}
	)
	.AddObjModel(
		ObjModel(OBJ2_PATH, TGA2_PATH),
		{
			2, 0, 0, 0,
			0, 2, 0, 0,
			0, 0, 2, 0,
			0, 0, 0, 1
		}
	).Commit();
	WinGDI_Window(hInstance, world, WINDOW_LENGTH, WINDOW_LENGTH)
		//.AddProcedure(MovePositionProcedure(0.05, PI / 256))
		.AddProcedure(
			UpdateProcedure(UpdateFunc, WINDOW_LENGTH, WINDOW_LENGTH)
		).Show().MessageLoop();
}



//////////////////////////////////////////////////////

int main()

{
	HINSTANCE instance = GetModuleHandle("MyRenderer.exe");
	// set the log level and is_print
	Log::SetLevel(Log$::DEBUG_LEVEL);
	Log::Info(LOG_NAME, "%p: May you have a nice day!", instance);

	StartRender(instance);

	return 0;
}
