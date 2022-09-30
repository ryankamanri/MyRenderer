#include <thread>
#include <map>
#include "kamanri/utils/logs.hpp"
#include "kamanri/windows/window.hpp"

using namespace Kamanri::Utils;
using namespace Kamanri::Windows;


namespace Kamanri
{   
    namespace Windows
    {
        namespace __Window
        {
            int WINDOW_WIDTH;
            int WINDOW_HEIGHT;

            const bool IS_PRINT = false;
        } // namespace __Window
        
    } // namespace Windows
    
    
} // namespace Kamanri




LRESULT CALLBACK WindowProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

/**
 * @brief the map used to find the window by the certain handle
 *
 */
std::map<HWND, Window *> window_map;

MSG msg;

Window::Window(HINSTANCE h_instance, int window_width, int window_height)
{
    __Window::WINDOW_WIDTH = window_width;
    __Window::WINDOW_HEIGHT = window_height;
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
        __Window::WINDOW_WIDTH, __Window::WINDOW_HEIGHT,                          //窗口的宽和高
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
    for (auto i = begin; i != window_map.end(); i++)
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
    if(__Window::IS_PRINT) PrintLn("Thread Id: %d | hWnd: %d | uMsg: %3d | wParam: %6X | lParam: %8X", tid, hWnd, uMsg, wParam, lParam);

    // the size of window_map equals 0 means no window is initialized.
    if (window_map.size() != 0)
    {
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
    }

    return DefWindowProc(hWnd, uMsg, wParam, lParam); //默认的窗口处理函数
}

void Window::_Paint()
{
    _paint_thread = std::thread([this]
    {
        DrawFunc(PainterFactor(_h_wnd, __Window::WINDOW_WIDTH, __Window::WINDOW_HEIGHT));
    });
}


