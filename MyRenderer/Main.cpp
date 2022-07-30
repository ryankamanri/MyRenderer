// CMakeProject1.cpp: 定义应用程序的入口点。
//
#include <iostream>
#include <thread>
#include <time.h>
#include "kamanri/utils/logs.h"
#include "kamanri/windows/windows.h"
#include "kamanri/utils/result.h"
#include "kamanri/maths/matrix.h"
#include "kamanri/maths/vectors.h"
#include "kamanri/renderer/obj_reader.h"
#include "kamanri/renderer/cameras.h"
#include "kamanri/renderer/world3ds.h"

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
		while (true)
		{
			for (int i = 0; i < 200; i += 3)
			{
				for (int j = 0; j < i; j++)
				{
					for (int k = 0; k < i; k++)
					{
						painter.Dot(j, k, RGB(i, j, k));
					}
				}
				painter.Flush();
			}
			for (int i = 0; i < 200; i += 3)
			{
				for (int j = 0; j < i; j++)
				{
					for (int k = 0; k < i; k++)
					{
						painter.Dot(j, k, RGB(j, k, i));
					}
				}
				painter.Flush();
			}
			for (int i = 0; i < 200; i += 3)
			{
				for (int j = 0; j < i; j++)
				{
					for (int k = 0; k < i; k++)
					{
						painter.Dot(j, k, RGB(k, i, j));
					}
				}
				painter.Flush();
			}

		}
	};
	window->DrawFunc = func;
	window->Show();
	Window::MessageLoop();
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPreInstance, LPSTR lpCmdLine, int nCmdShow)
{

	auto model = ObjModel();
	model.Read("./out/skybox.obj");
	auto v = **model.GetVertice(3);
	PrintLn("%f, %f, %f", v[0], v[1], v[2]);

	Camera camera({0, 2, 3, 1}, {0, -1, -1, 0}, {0, 1, 0, 0});

	World3D world(model, camera);

	system("pause");
	return 0;
}
