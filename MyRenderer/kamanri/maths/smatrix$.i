%module SMatrixE
%feature("python:annotations", "c");
%{
#include "kamanri/maths/smatrix$.hpp"
%}

%include "kamanri/maths/smatrix$.hpp"