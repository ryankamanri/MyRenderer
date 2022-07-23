// CMakeProject1.cpp: 定义应用程序的入口点。
//
#include <iostream>
#include <thread>
#include <time.h>
#include "kamanri/utils/logs.h"
#include "kamanri/windows/windows.h"
#include "kamanri/renderer/obj_reader.h"
#include "kamanri/utils/result.h"
#include "kamanri/maths/matrix.h"
#include "kamanri/maths/vectors.h"

using namespace Kamanri::Utils::Logs;
using namespace Kamanri::Utils::Memory;
using namespace Kamanri::Utils::Result;
using namespace Kamanri::Utils::Thread;
using namespace Kamanri::Windows::Windows;
using namespace Kamanri::Renderer::ObjReader;

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
	model.Read("./out/floor.obj");
	system("pause");
	return 0;
}
