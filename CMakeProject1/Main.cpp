// CMakeProject1.cpp: 定义应用程序的入口点。
//
#include <iostream>
#include <thread>
#include <time.h>
#include "mylibs/utils/log.h"
#include "mylibs/window/window.h"
#include "mylibs/utils/result.h"
#include "mylibs/maths/matrix.h"
#include "mylibs/maths/vector.h"

SOURCE_FILE("../Main.cpp");
constexpr const char *LOG_NAME = "Main";


//绘制方块
void Draw(Painter painter, int n, COLORREF color = 0);

//入口函数：所有代码都从这里开始执行
// WinMain:C语言Windows窗口程序入口函数

//做游戏窗口的步骤
// 1.设计窗口类
// 2.注册窗口类
// 3.创建窗口
// 4.显示窗口
// 5.更新窗口
// 6.消息循环

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPreInstance, LPSTR lpCmdLine, int nCmdShow)
{
	
	auto window = New<Window>(hInstance);

	auto func = [](Painter painter)
	{
		while (true)
		{
			for (int i = 0; i < 200; i += 3)
			{
				Draw(painter, i, RGB(0, 0xff, 0xff));
				painter.Flush();
			}

			Draw(painter, 600);
			painter.Flush();
		}
	};
	window->DrawFunc = func;
	window->Show();
	window->Update();
	window->MessageLoop();
	return 0;
}


// //绘制方块
void Draw(Painter painter, int n, COLORREF color)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			painter.Dot(i, j, color);
		}
	}
}
