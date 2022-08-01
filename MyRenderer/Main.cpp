// CMakeProject1.cpp: 定义应用程序的入口点。
//
#include <iostream>
#include <thread>
#include <time.h>
#include <float.h>
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
using namespace Kamanri::Windows::Windows;
using namespace Kamanri::Renderer::ObjReader;
using namespace Kamanri::Renderer::Cameras;
using namespace Kamanri::Renderer::World3Ds;


SOURCE_FILE("../Main.cpp");
constexpr const char *LOG_NAME = "Main";


void OpenWindow(HINSTANCE hInstance)
{
	
	auto window = New<Window>(hInstance);

	auto func = [](Painter painter)
	{
		auto model = ObjModel();
		model.Read("./out/skybox.obj");

		Camera camera({0, 2, 3, 1}, {0, -1, -1, 0}, {0, 1, 0, 0}, -1, -10, 600, 600);

		World3D world = World3D(model, camera);

		camera.Transform();

		world.Build();

		for(auto i = 0; i < 600; i++)
		{
			for(auto j = 0; j < 600; j++)
			{
				double depth = world.Depth(i, j);
				int color = -(255 / depth);
				if(depth != -DBL_MAX) Log::Trace(LOG_NAME, "x: %d, y: %d, depth: %f, color: %d", i, j, depth, color);
				painter.Dot(i, j, RGB(color, color, color));
			}
		}
	};

	window->DrawFunc = func;
	window->Show();
	Window::MessageLoop();
}



int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPreInstance, LPSTR lpCmdLine, int nCmdShow)
{

	OpenWindow(hInstance);

	system("pause");
	return 0;
}
