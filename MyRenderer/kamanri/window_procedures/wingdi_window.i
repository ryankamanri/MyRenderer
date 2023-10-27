%module(directors="1") WinGDI_Window

%{
#include "kamanri/utils/result.hpp"
#include "kamanri/utils/delegate.hpp"
#include "kamanri/window_procedures/wingdi_window/move_position_procedure.hpp"
#include "kamanri/window_procedures/wingdi_window/update_procedure.hpp"
%}


%include "kamanri/window_procedures/wingdi_window/move_position_procedure.hpp"
%include "kamanri/window_procedures/wingdi_window/update_procedure.hpp"

%include "kamanri/utils/delegate.hpp"
%template(WinGDI_Message_Delegate) Kamanri::Utils::Delegate<Kamanri::Windows::WinGDI_Window$::WinGDI_Message>;


// Kamanri::Utils::Result
namespace Kamanri
{
	namespace Utils
	{
		template<class T>
		class Result{};
	}
}

%template(DefaultResult) Kamanri::Utils::Result<void*>;


%feature("director") UpdateFuncBaseWrapper;
%inline %{
	struct UpdateFuncBaseWrapper
	{
		UpdateFuncBaseWrapper() {}
		virtual int UpdateFunc(Kamanri::Renderer::World::World3D& world) = 0;
		~UpdateFuncBaseWrapper() {}
	};
%}

%{
	static UpdateFuncBaseWrapper* update_func_base_wrapper_mount;
	static int UpdateFuncBaseWrapperCaller(Kamanri::Renderer::World::World3D& world)
	{
		auto result_code = update_func_base_wrapper_mount->UpdateFunc(world);
		return result_code;
	}
%}

%inline %{
	Kamanri::WindowProcedures::WinGDI_Window::UpdateProcedure 
	MakeUpdateProcedure(
		UpdateFuncBaseWrapper* wrapper, unsigned int screen_width, unsigned int screen_height)
	{
		update_func_base_wrapper_mount = wrapper;
		return Kamanri::WindowProcedures::WinGDI_Window::UpdateProcedure(UpdateFuncBaseWrapperCaller, screen_width, screen_height);
	}
%}