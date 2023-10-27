%module WinGDI_WindowE
%feature("python:annotations", "c");

%{
#include "kamanri/windows/wingdi_window$.hpp"
%}

%include <windows.i>

using HWND = void*;
using DWORD = unsigned long;

%include "kamanri/windows/wingdi_window$.hpp"