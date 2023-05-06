%module Renderer
%feature("python:annotations", "c");

%{
#include "kamanri/renderer/obj_model.hpp"
#include "kamanri/renderer/tga_image.hpp"
%}

%include <std_string.i>

%include "kamanri/renderer/obj_model.hpp"
%include "kamanri/renderer/tga_image.hpp"