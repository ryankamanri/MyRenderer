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

SOURCE_FILE("../Main.cpp");
constexpr const char *LOG_NAME = "Main";

void DrawFunc(PainterFactor painter_factor)
{
	auto painter = painter_factor.CreatePainter();

	auto jet = ObjModel("./out/jet.obj");
	auto floor = ObjModel("./out/floor.obj");

	auto camera = Camera({0, 0.5, -3, 1}, {0, 0, 1, 0}, {0, 1, 0, 0}, -1, -10, 600, 600);

	auto world = World3D(camera);

	auto floor_obj = **world.AddObjModel(floor);
	auto jet_obj = **world.AddObjModel(jet);

	SMatrix bigger_jet = 
	{
		4, 0, 0, 0,
		0, 4, 0, 0,
		0, 0, 4, 0,
		0, 0, 0, 1
	};

	jet_obj.Transform(bigger_jet);


	// revolve matrix
	double theta = M_PI / 24;
	SMatrix revolve_matrix =
		{
			1, 0, 0, 0,
			0, cos(theta), -sin(theta), 0,
			0, sin(theta), cos(theta), 0,
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

		double min_width;
		double min_height;
		double max_width;
		double max_height;

		world.GetMinMaxWidthHeight(min_width, min_height, max_width, max_height);

		auto min_width_int = (int)min_width;
		auto min_height_int = (int)min_height;
		auto max_width_int = (int)max_width;
		auto max_height_int = (int)max_height;



		Log::Debug(LOG_NAME, "Start to render...");

		int color;

		for (int i = min_width_int; i <= max_width_int; i++)
		{

			if (i > 600 || i < 0)
				return;
			for (int j = min_height_int; j <= max_height_int && j <= 600 && j >= 0; j++)
			{
				if (j > 600 || j < 0)
					continue;

				color = -(int)(255 / (world.Depth(i, j) / 5));

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

	auto window = New<Window>(hInstance);

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
