%module World
%feature("python:annotations", "c");

%{
#include "kamanri/utils/result.hpp"
#include "kamanri/renderer/world/blinn_phong_reflection_model.hpp"
#include "kamanri/renderer/world/camera.hpp"
#include "kamanri/renderer/world/frame_buffer.hpp"
#include "kamanri/renderer/world/object.hpp"
#include "kamanri/renderer/world/world3d.hpp"
%}

%include "kamanri/utils/result.hpp"
%include "kamanri/types.i.hpp"
%template(DefaultResult) Kamanri::Utils::Result<void*>;
using DWORD = unsigned long;
%template(__BlinnPhongReflectionModelE_PointLightList) std::vector<Kamanri::Renderer::World::BlinnPhongReflectionModel$::PointLight>;

%include "kamanri/renderer/world/blinn_phong_reflection_model.hpp"
%include "kamanri/renderer/world/camera.hpp"
%include "kamanri/renderer/world/frame_buffer.hpp"
%include "kamanri/renderer/world/object.hpp"
%include "kamanri/renderer/world/world3d.hpp"