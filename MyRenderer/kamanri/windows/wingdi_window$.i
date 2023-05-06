%module WinGDI_WindowE
%feature("python:annotations", "c");

%{
#include "kamanri/windows/wingdi_window$.hpp"
%}

%include <windows.i>

%include "kamanri/windows/wingdi_window$.hpp"