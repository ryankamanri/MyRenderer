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
#include "kamanri/maths/matrix.hpp"
#include "kamanri/maths/vectors.hpp"
#include "kamanri/renderer/obj_reader.hpp"
#include "kamanri/renderer/cameras.hpp"
#include "kamanri/renderer/world3ds.hpp"

using namespace Kamanri::Utils::Logs;
using namespace Kamanri::Utils::Memory;
using namespace Kamanri::Utils::Result;
using namespace Kamanri::Utils::Thread;
using namespace Kamanri::Maths::Vectors;
using namespace Kamanri::Maths::Matrix;
using namespace Kamanri::Windows::Windows;
using namespace Kamanri::Renderer::ObjReader;
using namespace Kamanri::Renderer::Cameras;
using namespace Kamanri::Renderer::World3Ds;

SOURCE_FILE("../Main.cpp");
constexpr const char *LOG_NAME = "Main";

void DrawFunc(Painter painter)
{
	auto model = ObjModel();
	model.Read("./out/skybox.obj");

	// Camera camera({0, 2, 3, 1}, {0, -1, -1, 0}, {0, 1, 0, 0}, -1, -10, 600, 600);
	Camera camera({0, 0, 5, 1}, {0, -0.2, -1, 0}, {0, 1, 0, 0}, -1, -10, 600, 600);

	World3D world = World3D(model, camera);

	// revolve matrix
	double theta = M_PI / 12;
	SMatrix revolve_matrix =
		{
			cos(theta), 0, -sin(theta), 0,
			0, 1, 0, 0,
			sin(theta), 0, cos(theta), 0,
			0, 0, 0, 1
		};

	while (true)
	{
		Log::Trace(LOG_NAME, "Direction transform: ");
		camera.GetDirection().PrintVector();
		revolve_matrix * camera.GetDirection();
		camera.GetDirection().PrintVector();
		Log::Trace(LOG_NAME, "Location transform: ");
		camera.GetLocation().PrintVector();
		revolve_matrix * camera.GetLocation();
		camera.GetLocation().PrintVector();

		//
		camera.Transform();

		world.Build();

		double min_width;
		double min_height;
		double max_width;
		double max_height;

		world.GetMinMaxWidthHeight(min_width, min_height, max_width, max_height);

		auto min_width_int = (int)min_width;
		auto min_height_int = (int)min_height;
		auto max_width_int = (int)max_width;
		auto max_height_int = (int)max_height;

		double depth;
		int color;

		for (int i = min_width_int; i <= max_width_int; i++)
		{
			if(i > 600 || i < 0) continue;
			for (int j = min_height_int; j <= max_height_int && j <= 600 && j >= 0; j++)
			{
				if(j > 600 || j < 0) continue;
				depth = world.Depth(i, j);
				color = -(int)(255 / (depth / 5));
				// if(depth != -DBL_MAX) Log::Trace(LOG_NAME, "x: %d, y: %d, depth: %f, color: %d", i, j, depth, color);
				painter.Dot(i, j, RGB(color, color, color));
			}
			painter.Flush();
		}

		Log::Info(LOG_NAME, "Finish a frame render.");
	}
}

void OpenWindow(HINSTANCE hInstance)
{

	auto window = New<Window>(hInstance);

	window->DrawFunc = DrawFunc;
	window->Show();
	Window::MessageLoop();
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPreInstance, LPSTR lpCmdLine, int nCmdShow)
{

	OpenWindow(hInstance);

	system("pause");
	return 0;
}
