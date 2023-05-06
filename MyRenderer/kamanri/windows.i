%module Windows
%feature("python:annotations", "c");

%{
#include "kamanri/windows/wingdi_window.hpp"
%}

%include <windows.i>
%include "kamanri/types.i.hpp"

void* IntToPtr(unsigned long long value)
{
	return (void*)value;
}

%include "kamanri/windows/wingdi_window.hpp"

%include "kamanri/utils/delegate.hpp"

%template(__WinGDI_Message_Delegate) Kamanri::Utils::Delegate<Kamanri::Windows::WinGDI_Window$::WinGDI_Message>;


