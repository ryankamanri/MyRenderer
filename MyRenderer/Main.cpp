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

constexpr const char* OBJ_PATH = "../../out/jet.obj";


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
} // namespace __UpdateFunc

DefaultResult UpdateFunc(World3D& world)
{
	using namespace __UpdateFunc;
	Camera& camera = world.GetCamera();
	direction = camera.Direction();

	revolve_matrix* camera.Direction();
	revolve_matrix* camera.Location();

	camera.InverseUpperWithDirection(direction);

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
					{ 0, 0.5, -3, 1 },
					{ 0, 0, 1, 0 },
					{ 0, 1, 0, 0 },
					-1,
					-10,
					WINDOW_LENGTH,
					WINDOW_LENGTH
				)
			).AddObjModel(
				ObjModel(OBJ_PATH),
				{ 3, 0, 0, 0,
				 0, 3, 0, 0,
				 0, 0, 3, 0,
				 0, 0, 0, 1 }
			)).AddProcedure(
				UpdateProcedure(UpdateFunc, WINDOW_LENGTH, WINDOW_LENGTH)
			).Show().MessageLoop();
}



void ResourcePoolTest()
{
	ResourcePool<SMatrix, 2> pool([]()
	{
		return Result<SMatrix>(SMatrix(3));
	});

	auto item = pool.Allocate();
	item.data.PrintMatrix();

	auto item2 = pool.Allocate();
	item2.data.PrintMatrix();

	pool.Free(item);
}
//////////////////////////////////////////////////////

int WinMain(HINSTANCE hInstance, HINSTANCE hPreInstance, LPSTR lpCmdLine, int nCmdShow)
{
	Log::Level(Log$::WARN_LEVEL);
	StartRender(hInstance);
	// ResourcePoolTest();
	system("pause");
	return 0;
}
