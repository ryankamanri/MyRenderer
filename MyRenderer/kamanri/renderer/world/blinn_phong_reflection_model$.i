%module BlinnPhongReflectionModelE

%{
#include "kamanri/renderer/world/blinn_phong_reflection_model$.hpp"
%}

%include "kamanri/renderer/world/blinn_phong_reflection_model$.hpp"

%include <std_vector.i>

%template(PointLightList) std::vector<Kamanri::Renderer::World::BlinnPhongReflectionModel$::PointLight>;