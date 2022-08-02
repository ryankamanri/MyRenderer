#include <thread>
#include <map>
#include "../../utils/logs.hpp"
#include "../../windows/windows.hpp"

using namespace Kamanri::Utils::Logs;
using namespace Kamanri::Windows::Windows;

const int DEFAULT_LENGTH = 600;

const int WINDOW_WIDTH = DEFAULT_LENGTH;
const int WINDOW_HEIGHT = DEFAULT_LENGTH;

const bool IS_PRINT = false;

LRESULT CALLBACK WindowProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

/**
 * @brief the map used to find the window by the certain handle
 *
 */
std::map<HWND, Window *> window_map;

MSG msg;

Window::Window(HINSTANCE h_instance)
{
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
        200, 100,                                             //窗口左上角坐标（像素）
        WINDOW_WIDTH, WINDOW_HEIGHT,                          //窗口的宽和高
        NULL,                                                 //父窗口句柄
        NULL,                                                 //菜单句柄
        h_instance,                                           //应用程序实例句柄
        NULL                                                  //附加参数
    );

    window_map.insert(std::pair<HWND, Window *>(_h_wnd, this));
}

Window::~Window()
{
    auto begin = window_map.begin();
    auto end = window_map.end();
    for (auto i = begin; i != end; i++)
    {
        if(i->second == this)
        {
            window_map.erase(i->first);
            return;
        }
    }
}

bool Window::Show()
{
    return ShowWindow(_h_wnd, SW_SHOW);
}

bool Window::Update()
{
    return UpdateWindow(_h_wnd);
}

void Window::MessageLoop()
{

    while (GetMessage(&msg, NULL, 0, 0)) // GetMessage从调用线程的消息队列中取得一个消息并放于msg
    {
        //将虚拟键消息转换为字符消息
        TranslateMessage(&msg);
        //将消息分发给窗口处理函数
        DispatchMessage(&msg);
    }
}

// //窗口处理函数
LRESULT CALLBACK WindowProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    auto tid = GetCurrentThreadId();
    if(IS_PRINT) PrintLn("Thread Id: %d | hWnd: %d | uMsg: %3d | wParam: %6X | lParam: %8X", tid, hWnd, uMsg, wParam, lParam);

    auto p_window = window_map.find(hWnd)->second;

    switch (uMsg)
    {
    case WM_PAINT: //窗口绘图消息
        p_window->_Paint();
        break;
    case WM_CLOSE: //窗口关闭消息
        DestroyWindow(hWnd);
        break;
    case WM_DESTROY: //窗口销毁消息
        PostQuitMessage(0);
        break;
    }
    return DefWindowProc(hWnd, uMsg, wParam, lParam); //默认的窗口处理函数
}

void Window::_Paint()
{
    _paint_thread = std::thread([this]
    {
        DrawFunc(PainterFactor(_h_wnd));
    });
}



PainterFactor::PainterFactor(HWND h_wnd): _h_wnd(h_wnd)
{
    _h_dc = GetDC(_h_wnd);

    _h_black_dc = CreateCompatibleDC(_h_dc);
    HBITMAP hBackBmp = CreateCompatibleBitmap(_h_dc, WINDOW_WIDTH, WINDOW_HEIGHT);
    SelectObject(_h_black_dc, hBackBmp);
}

PainterFactor::~PainterFactor()
{
    ReleaseDC(_h_wnd, _h_dc);
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
    HBITMAP hBackBmp = CreateCompatibleBitmap(_h_dc, WINDOW_WIDTH, WINDOW_HEIGHT);

    SelectObject(h_draw_dc, hBackBmp);

    return Painter(_h_dc, h_draw_dc);
}

void PainterFactor::Clean(Painter& painter)
{
    BitBlt(painter._h_draw_dc, 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, _h_black_dc, 0, 0, SRCCOPY);
}


Painter::Painter(HDC h_dc, HDC h_mem_dc): _h_dc(h_dc), _h_draw_dc(h_mem_dc) {}

Painter::~Painter()
{
    DeleteObject(_h_draw_dc);
}

WINBOOL Painter::Flush() const { return BitBlt(_h_dc, 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, _h_draw_dc, 0, 0, SRCCOPY); }