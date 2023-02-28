#include <thread>
#include <map>
#include "kamanri/utils/logs.hpp"
#include "kamanri/windows/wingdi_window.hpp"

using namespace Kamanri::Utils;
using namespace Kamanri::Windows;
using namespace Kamanri::Windows::WinGDI_Window$;

namespace Kamanri
{   
	namespace Windows
	{
		

		namespace __Window
		{
			
			const bool IS_PRINT = false;
		} // namespace __Window

		namespace __Painter
		{
			
		} // namespace __Painter

		namespace __PainterFactor
		{

		} // namespace __PainterFactor
		
	} // namespace Windows
	
	
} // namespace Kamanri




LRESULT CALLBACK WindowProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

/**
 * @brief the map used to find the window by the certain handle
 *
 */
std::map<HWND, WinGDI_Window *> window_map;

MSG msg;

WinGDI_Window::WinGDI_Window(HINSTANCE h_instance, Renderer::World::World3D& world, unsigned int window_width, unsigned int window_height)
: _world(world)
{
	_window_width = window_width;
	_window_height = window_height;
	_WindowProc = WindowProc;
	// 1.设计窗口类
	TCHAR szAppClassName[] = TEXT("ZWX");
	WNDCLASS wc = {0};
	wc.hbrBackground = CreateSolidBrush(RGB(0, 0, 0)); //背景颜色画刷
	wc.hCursor = LoadCursor(NULL, IDC_HAND);           //鼠标光标类型,手：DC_HAND
	wc.hIcon = LoadIcon(NULL, IDI_APPLICATION);        //图标
	wc.hInstance = h_instance;                         //应用程序实例句柄，表示exe
	wc.lpfnWndProc = _WindowProc;                      //窗口处理函数
	wc.lpszClassName = szAppClassName;                 //窗口类型名
	wc.style = CS_HREDRAW | CS_VREDRAW;                //窗口类的风格

	// 2.注册窗口类
	RegisterClass(&wc);

	// 3.创建窗口, and bind this window object with handle
	_h_wnd = CreateWindow(
		szAppClassName,                                       //窗口类型名
		TEXT("Graphic"),                                      //窗口标题
		WS_BORDER | WS_CAPTION | WS_MINIMIZEBOX | WS_SYSMENU, //窗口的风格
		0, 0,                                             //窗口左上角坐标（像素）
		_window_width, _window_height,                          //窗口的宽和高
		NULL,                                                 //父窗口句柄
		NULL,                                                 //菜单句柄
		h_instance,                                           //应用程序实例句柄
		NULL                                                  //附加参数
	);

	window_map.insert(std::pair<HWND, WinGDI_Window *>(_h_wnd, this));
	
}

WinGDI_Window::~WinGDI_Window()
{
	auto begin = window_map.begin();
	for (auto i = begin; i != window_map.end(); i++)
	{
		if(i->second == this)
		{
			window_map.erase(i->first);
			return;
		}
	}
}


WinGDI_Window& WinGDI_Window::AddProcedure(Delegate<WinGDI_Message>::ANode&& proc)
{
	this->_procedure_chain.AddRear(proc);
	return *this;
}

WinGDI_Window& WinGDI_Window::Show()
{
	ShowWindow(_h_wnd, SW_SHOW);
	return *this;
}

WinGDI_Window& WinGDI_Window::Update()
{
	UpdateWindow(_h_wnd);
	return *this;
}

void WinGDI_Window::MessageLoop()
{
	// 应用程序在运行时将收到数千条消息。（请考虑每次击键和鼠标按钮单击都会生成一条消息。
	// 此外，应用程序可以有多个窗口，每个窗口都有自己的窗口过程。
	// 程序如何接收所有这些消息并将它们传递到正确的窗口过程？应用程序需要一个循环来检索消息并将其调度到正确的窗口。

	// 对于创建窗口的每个线程，操作系统都会为窗口消息创建一个队列。此队列保存在该线程上创建的所有窗口的消息。
	// 队列本身对程序是隐藏的。不能直接操作队列。但是，您可以通过调用 GetMessage 函数从队列中提取消息。

	while (GetMessage(&msg, NULL, 0, 0)) // GetMessage从调用线程的消息队列中取得一个消息并放于msg
	{
		//将虚拟键消息转换为字符消息
		TranslateMessage(&msg);
		//将消息分发给窗口处理函数
		DispatchMessage(&msg);
		
	}
}

// //窗口处理函数
LRESULT CALLBACK Kamanri::Windows::WindowProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{ 
	if(__Window::IS_PRINT)
	{
		auto tid = GetCurrentThreadId();
		PrintLn("Thread Id: %d | hWnd: %d | uMsg: %3d | wParam: %6X | lParam: %8X", tid, hWnd, uMsg, wParam, lParam);
	} 

	// the size of window_map equals 0 means no window is initialized.
	if (window_map.size() != 0)
	{
		WinGDI_Window& window = *window_map[hWnd];

		// Build a message call system refer to 'RequestDelegate', pluginize like middleware.
		window._procedure_chain.Execute(WinGDI_Window$::WinGDI_Message(window._world, hWnd, uMsg, wParam, lParam));

		switch (uMsg)
		{
		case WM_CLOSE: //窗口关闭消息
			DestroyWindow(hWnd);
			break;
		case WM_DESTROY: //窗口销毁消息
			PostQuitMessage(0);
			break;
		}
	}

	return DefWindowProc(hWnd, uMsg, wParam, lParam); //默认的窗口处理函数
}






PainterFactor::PainterFactor(HWND h_wnd, unsigned int window_width, unsigned int window_height): _h_wnd(h_wnd)
{
	_window_width = window_width;
	_window_height = window_height;

	_h_dc = GetDC(_h_wnd);

	_h_black_dc = CreateCompatibleDC(_h_dc);
	HBITMAP hBackBmp = CreateCompatibleBitmap(_h_dc, _window_width, _window_height);
	SelectObject(_h_black_dc, hBackBmp);
}

PainterFactor::~PainterFactor()
{
	ReleaseDC(_h_wnd, _h_dc);
}

void PainterFactor::Clean(Painter& painter)
{
	BitBlt(painter._h_mem_dc, 0, 0, _window_width, _window_height, _h_black_dc, 0, 0, SRCCOPY);
}

Painter PainterFactor::CreatePainter()
{
	// on paint
	//创建内存DC（先放到内存中）
	HDC h_draw_dc = CreateCompatibleDC(_h_dc);

	//创建一张兼容位图
	// note:
	//这要注意,如果创建和内存DC兼容的位图就只有黑白色,不会有彩色
	//所以要创建实际对象DC.窗口DC或静态控件DC兼容的内存位图
	HBITMAP h_bitmap = CreateCompatibleBitmap(_h_dc, _window_width, _window_height);

	

	SelectObject(h_draw_dc, h_bitmap);

	return Painter(_h_dc, h_draw_dc, h_bitmap, _window_width, _window_height);
}



Painter::Painter(HDC h_dc, HDC h_mem_dc, HBITMAP h_bitmap, unsigned int window_width, unsigned int window_height) : _h_dc(h_dc), _h_mem_dc(h_mem_dc) , _h_bitmap(h_bitmap)
{
	_window_width = window_width;
	_window_height = window_height;

	// set bitmap info
	_b_info.bmiHeader.biSize = 40;
	_b_info.bmiHeader.biWidth = _window_width;
	_b_info.bmiHeader.biHeight = _window_height;
	_b_info.bmiHeader.biPlanes = 1;
	_b_info.bmiHeader.biBitCount = 8 * sizeof(DWORD);
	_b_info.bmiHeader.biCompression = BI_RGB;
	//
}

Painter::~Painter()
{
	DeleteObject(_h_mem_dc);
}

BOOL Painter::Flush() const 
{ 
	return BitBlt(_h_dc, 0, 0, _window_width, _window_height, _h_mem_dc, 0, 0, SRCCOPY); 
}

int Painter::DrawFrom(DWORD* bitmap)
{
	return StretchDIBits(
		_h_mem_dc,
		0,
		0,
		_window_width,
		_window_height,
		0,
		0,
		_window_width,
		_window_height,
		(void *)bitmap,
		&_b_info,
		DIB_RGB_COLORS,
		SRCCOPY
	);
}
