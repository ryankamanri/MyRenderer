// CMakeProject1.cpp: 定义应用程序的入口点。
//
#include <iostream>
#include <thread>
#include <cfloat>
#include <cmath>
#include "kamanri/all.hpp"

using namespace Kamanri::Utils;
using namespace Kamanri::Maths;
using namespace Kamanri::Windows;
using namespace Kamanri::Windows::WinGDI_Window$;
using namespace Kamanri::Renderer;
using namespace Kamanri::Renderer::World;


constexpr const char *LOG_NAME = "Main";
const int WINDOW_LENGTH = 600;

constexpr const char *OBJ_PATH = "../../out/jet.obj";

void DrawFunc(PainterFactor painter_factor)
{
	auto painter = painter_factor.CreatePainter();

	// in vscode
	// auto j20 = ObjModel("./out/j20.obj");
	auto floor = ObjModel(OBJ_PATH);

	// in vs
	// auto j20 = ObjModel("../../j20.obj");
	// auto floor = ObjModel("../../floor.obj");

	auto camera = Camera({0, 0.5, -3, 1}, {0, 0, 1, 0}, {0, 1, 0, 0}, -1, -10, WINDOW_LENGTH, WINDOW_LENGTH);

	auto world = World3D(camera);

	auto floor_obj = *world.AddObjModel(floor);

	floor_obj.Transform
	({
		2, 0, 0, 0,
		0, 2, 0, 0,
		0, 0, 2, 0,
		0, 0, 0, 1
	});

	// auto j20_obj = *world.AddObjModel(j20);

	// TGAImage img{};
	// img.ReadTGAFile("out/j20.tga");

	// SMatrix bigger_j20 = 
	// {
	// 	0.01, 0, 0, 0,
	// 	0, 0, 0.01, 0,
	// 	0, 0.01, 0, 0,
	// 	0, 0, 0, 1
	// };

	// j20_obj.Transform(bigger_j20);


	// revolve matrix
	double theta = PI / 64;
	SMatrix revolve_matrix =
		{
			cos(theta), 0, -sin(theta), 0,
			0, 1, 0, 0,
			sin(theta), 0, cos(theta), 0,
			0, 0, 0, 1
		};

	while (true)
	{
		// this part rightly belongs to DrawFunc
		auto direction = camera.Direction();

		revolve_matrix * camera.Direction();
		revolve_matrix * camera.Location();

		camera.InverseUpperWithDirection(direction);
		
		camera.Transform();

		world.Build();
		//
		Log::Info(LOG_NAME, "Start to render...");

		for (int i = 0; i <= WINDOW_LENGTH; i++)
		{

			for (int j = 0; j <= WINDOW_LENGTH; j++)
			{
				auto depth = world.Depth(i, j);
				if(depth == -DBL_MAX) continue;

				int color = -(int)(255 / (depth / 1500)); // depth range [1500, inf)

				painter.Dot(i, j, RGB(color, color, color));

				// auto color = img.Get(i, j);
				// painter.Dot(i, j, color.bgr);
			}
		}
		
		painter.Flush();
		painter_factor.Clean(painter);
		Log::Info(LOG_NAME, "Finish a frame render.");
	}
}

void OpenWindow(HINSTANCE hInstance)
{

	WinGDI_Window window(hInstance, WINDOW_LENGTH, WINDOW_LENGTH);

	window.DrawFunc = DrawFunc;
	window.Show();
	WinGDI_Window::MessageLoop();



	// TODO:
	// CameraAttributes c_attr = CameraAttributes(attrs...);
	// AWindow window = WinGDI_Window(attrs); // extend the abstract class Kamanri::Renderer::World::Camera::AWindow 
	// window.AddProcedure(RecursiveProcedure(RecursiveFunc))
	// 			.AddProcedure(KeyboardControlProcedure);
	// auto camera = Camera(std::move(c_attr), std::move(window)); // rvalue window
	// wuto world = World3D(std::move(camera));
	// world.AddObjModel(model).ShowWindow().Run(); // Run calls AWindow::MessageLoop();
	//
	// Or:
	// World world(
	// 		Camera(
	// 			CameraAttributes(attrs...),
	// 			Window(attrs...).AddProcedure(RecursiveProcedure(RecursiveFunc)).AddProcedure(KeyboardControlProcedure)
	// 		)
	// ).AddObjModel(model).ShowWindow().Run();
	//
	// NOTE:
	// 1. Every class as a property of others should in form of rvalue reference. 
}

////////////////////////////////////////////////////
// delegate test





void DelegateTest()
{
	Delegate d;
	Delegate$::Node dn;
	Delegate$::Node dn2;
	dn.this_delegate = [](int res, Delegate$::Node& this_d_node)
	{
		PrintLn("%d", res);
		this_d_node.Next(res);
	};
	dn2.this_delegate = [](int res, Delegate$::Node &this_d_node)
	{
		PrintLn("Hello");
		this_d_node.Next(res);
		PrintLn("res: %d", res);
	};
	d.AddHead(dn);
	d.AddHead(dn2);
	d.Execute(1);
}

void ResourcePoolTest()
{
	ResourcePool<SMatrix, 2> pool([](){
		return Result<SMatrix>(SMatrix(3));
	});

	auto item = pool.Allocate();
	item.item.PrintMatrix();

	auto item2 = pool.Allocate();
	item2.item.PrintMatrix();

	pool.Free(item);


}
//////////////////////////////////////////////////////

int WinMain(HINSTANCE hInstance, HINSTANCE hPreInstance, LPSTR lpCmdLine, int nCmdShow)
{
	Log::Level(Log$::WARN_LEVEL);
	OpenWindow(hInstance);
	// DelegateTest();
	// ResourcePoolTest();
	system("pause");
	return 0;

}
