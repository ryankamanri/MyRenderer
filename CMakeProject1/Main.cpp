// CMakeProject1.cpp: 定义应用程序的入口点。
//
#include <iostream>
#include <thread>
#include <time.h>
#include "mylibs/mylog.h"
#include "renderers/tgaimage.h"
#include "renderers/model.h"
#include "renderers/line.h"
#include "mylibs/myresult.h"
#include "mylibs/mymatrix.h"
#include "mylibs/myvector.h"

SOURCE_FILE("../Main.cpp");
constexpr const char *LOG_NAME = "Main";

#include <windows.h>
#include <time.h>

const int DEFAULT_LENGTH = 600;

const int WINDOW_WIDTH = DEFAULT_LENGTH;
const int WINDOW_HEIGHT = DEFAULT_LENGTH;

//窗口处理函数
LRESULT CALLBACK WindowProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
// 绘图
void Paint(HWND hWnd);
void OnPaint(HDC hDC);
//绘制方块
void Draw(HDC hDC, int n,COLORREF color = 0);

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
    // 1.设计窗口类
    TCHAR szAppClassName[] = TEXT("ZWX");
    WNDCLASS wc = {0};
    wc.hbrBackground = CreateSolidBrush(RGB(0, 0, 0)); //背景颜色画刷
    wc.hCursor = LoadCursor(NULL, IDC_HAND);           //鼠标光标类型,手：DC_HAND
    wc.hIcon = LoadIcon(NULL, IDI_ERROR);              //图标
    wc.hInstance = hInstance;                          //应用程序实例句柄，表示exe
    wc.lpfnWndProc = WindowProc;                       //窗口处理函数
    wc.lpszClassName = szAppClassName;                 //窗口类型名
    wc.style = CS_HREDRAW | CS_VREDRAW;                //窗口类的风格

    // 2.注册窗口类
    RegisterClass(&wc);

    // 3.创建窗口
    HWND hWnd = CreateWindow(
        szAppClassName,                                       //窗口类型名
        TEXT("Graphic"),                                      //窗口标题
        WS_BORDER | WS_CAPTION | WS_MINIMIZEBOX | WS_SYSMENU, //窗口的风格
        200, 100,                                             //窗口左上角坐标（像素）
        WINDOW_WIDTH, WINDOW_HEIGHT,                                             //窗口的宽和高
        NULL,                                                 //父窗口句柄
        NULL,                                                 //菜单句柄
        hInstance,                                            //应用程序实例句柄
        NULL                                                  //附加参数
    );

    // 4.显示窗口
    ShowWindow(hWnd, SW_SHOW);

    // 5.更新窗口
    UpdateWindow(hWnd);

    // 6.消息循环
    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0)) // GetMessage从调用线程的消息队列中取得一个消息并放于msg
    {
        //将虚拟键消息转换为字符消息
        TranslateMessage(&msg);
        //将消息分发给窗口处理函数
        DispatchMessage(&msg);
    }
    return 0;
}

//窗口处理函数
LRESULT CALLBACK WindowProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{

    auto tid = GetCurrentThreadId();
    PrintLn("Thread Id: %d, Msg: %d", tid, uMsg);

    switch (uMsg)
    {
    case WM_PAINT: //窗口绘图消息
        Paint(hWnd);
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


// global scope thread
std::thread paint_thread;

void Paint(HWND hWnd)
{
    //开始绘图
    paint_thread = std::move(std::thread([hWnd]{
        HDC hDC;
        hDC = GetDC(hWnd);
        OnPaint(hDC);
        ReleaseDC(hWnd, hDC);
    }));
    
}

void OnPaint(HDC hDC)
{
    //创建内存DC（先放到内存中）
    HDC hMemDC = CreateCompatibleDC(hDC);
    //创建一张兼容位图
    // note:
    //这要注意,如果创建和内存DC兼容的位图就只有黑白色,不会有彩色
    //所以要创建实际对象DC.窗口DC或静态控件DC兼容的内存位图
    HBITMAP hBackBmp = CreateCompatibleBitmap(hDC, 600, 600);

    SelectObject(hMemDC, hBackBmp);

    // //绘制
    // Draw(hMemDC, 100);
    // //一次性绘制到界面上
    // BitBlt(hDC, 0, 0, 300, 600, hMemDC, 0, 0, SRCCOPY);

    while(true) 
    {
        for(int i = 0; i < 200; i+=3)
        {
            Draw(hMemDC, i, RGB(0, 0xff, 0xff));
            BitBlt(hDC, 0, 0, 600, 600, hMemDC, 0, 0, SRCCOPY);
            
        }
            
        Draw(hMemDC, 600);
        BitBlt(hDC, 0, 0, 600, 600, hMemDC, 0, 0, SRCCOPY);

        
    }

    //释放DC
    DeleteObject(hMemDC);
}



//绘制方块
void Draw(HDC hDC, int n, COLORREF color)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            SetPixel(hDC, i, j, color);
        }
    }

}


