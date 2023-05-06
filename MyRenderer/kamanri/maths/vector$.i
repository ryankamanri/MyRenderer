%module VectorE
%feature("python:annotations", "c");
%{
#include "kamanri/maths/vector$.hpp"
%}

%include "kamanri/maths/vector$.hpp"