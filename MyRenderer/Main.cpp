// CMakeProject1.cpp: 定义应用程序的入口点。
//
#include <iostream>
#include <thread>
#include <time.h>
#include <float.h>
#include <math.h>
#include "kamanri/utils/logs.hpp"
#include "kamanri/windows/windows.hpp"
#include "kamanri/utils/result.hpp"
#include "kamanri/utils/iterator.hpp"
#include "kamanri/maths/matrix.hpp"
#include "kamanri/maths/vectors.hpp"
#include "kamanri/renderer/obj_reader.hpp"
#include "kamanri/renderer/cameras.hpp"
#include "kamanri/renderer/world3ds.hpp"
#include "kamanri/utils/string.hpp"

using namespace Kamanri::Utils::Logs;
using namespace Kamanri::Utils::Memory;
using namespace Kamanri::Utils::Result;
using namespace Kamanri::Utils::Iterator;
using namespace Kamanri::Utils::Thread;
using namespace Kamanri::Maths::Vectors;
using namespace Kamanri::Maths::Matrix;
using namespace Kamanri::Windows::Windows;
using namespace Kamanri::Renderer::ObjReader;
using namespace Kamanri::Renderer::Cameras;
using namespace Kamanri::Renderer::World3Ds;

// SOURCE_FILE("../Main.cpp");
constexpr const char *LOG_NAME = "Main";
const int WINDOW_LENGTH = 600;

void DrawFunc(PainterFactor painter_factor)
{
	auto painter = painter_factor.CreatePainter();

	auto j20 = ObjModel("./out/j20.obj");
	auto floor = ObjModel("./out/floor.obj");

	auto camera = Camera({0, 0.5, -3, 1}, {0, 0, 1, 0}, {0, 1, 0, 0}, -1, -10, WINDOW_LENGTH, WINDOW_LENGTH);

	auto world = World3D(camera);

	auto floor_obj = **world.AddObjModel(floor);
	auto j20_obj = **world.AddObjModel(j20);

	SMatrix bigger_j20 = 
	{
		0.01, 0, 0, 0,
		0, 0, 0.01, 0,
		0, 0.01, 0, 0,
		0, 0, 0, 1
	};

	j20_obj.Transform(bigger_j20);


	// revolve matrix
	double theta = M_PI / 64;
	SMatrix revolve_matrix =
		{
			cos(theta), 0, -sin(theta), 0,
			0, 1, 0, 0,
			sin(theta), 0, cos(theta), 0,
			0, 0, 0, 1
		};

	while (true)
	{
		auto direction = *camera.Direction().Copy();

		revolve_matrix * camera.Direction();
		revolve_matrix * camera.Location();

		camera.InverseUpperWithDirection(direction);
		//
		camera.Transform();

		world.Build();

		Log::Info(LOG_NAME, "Start to render...");

		int color;

		for (int i = 0; i <= WINDOW_LENGTH; i++)
		{

			for (int j = 0; j <= WINDOW_LENGTH; j++)
			{
				auto depth = world.Depth(i, j);
				if(depth == -DBL_MAX) continue;

				color = -(int)(255 / (depth / 5));

				painter.Dot(i, j, RGB(color, color, color));
			}
		}
		
		painter.Flush();
		painter_factor.Clean(painter);
		Log::Info(LOG_NAME, "Finish a frame render.");
	}
}

void OpenWindow(HINSTANCE hInstance)
{

	auto window = New<Window>(hInstance, WINDOW_LENGTH, WINDOW_LENGTH);

	window->DrawFunc = DrawFunc;
	window->Show();
	Window::MessageLoop();
}




int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPreInstance, LPSTR lpCmdLine, int nCmdShow)
{
	Log::Level(INFO_LEVEL);
	OpenWindow(hInstance);

	
	

	system("pause");
	return 0;
}
