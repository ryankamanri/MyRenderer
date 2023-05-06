%module World
%feature("python:annotations", "c");

%{
#include "kamanri/renderer/world/blinn_phong_reflection_model.hpp"
#include "kamanri/renderer/world/camera.hpp"
#include "kamanri/renderer/world/frame_buffer.hpp"
#include "kamanri/renderer/world/object.hpp"
#include "kamanri/renderer/world/world3d.hpp"
%}

%include "kamanri/types.i.hpp"
%template(__BlinnPhongReflectionModel$_PointLightList) std::vector<Kamanri::Renderer::World::BlinnPhongReflectionModel$::PointLight>;

%include "kamanri/renderer/world/blinn_phong_reflection_model.hpp"
%include "kamanri/renderer/world/camera.hpp"
%include "kamanri/renderer/world/frame_buffer.hpp"
%include "kamanri/renderer/world/object.hpp"
%include "kamanri/renderer/world/world3d.hpp"