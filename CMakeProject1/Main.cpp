// CMakeProject1.cpp: 定义应用程序的入口点。
//
#include <iostream>
#include "mylibs/mylog.h"
#include "renderers/tgaimage.h"
#include "renderers/model.h"
#include "renderers/line.h"
#include "mylibs/myresult.h"
#include "mylibs/mymatrix.h"
#include "mylibs/myvector.h"

constexpr int width  = 800; // output image size
constexpr int height = 800;

const TGAColor white = TGAColor(255, 255, 255, 255);
const TGAColor red   = TGAColor(255, 0,   0,   255);
const TGAColor green = TGAColor(0, 255, 0, 255);
Model *model = NULL;

SOURCE_FILE("../Main.cpp");
constexpr const char* LOG_NAME = "Main";

void Triangle(Vec2i t0, Vec2i t1, Vec2i t2, TGAImage &image, TGAColor color) { 
    Line(t0, t1, image, color); 
    Line(t1, t2, image, color); 
    Line(t2, t0, image, color); 
}

void PrintVector(Vector v)
{
    v.PrintVector();
    Vector v2 = v;
    v2.PrintVector();
}

void MatrixTest()
{
    
    SMatrix sm = 
    {{1, 2, 3},
     {4, 5, 6},
     {7, 8, 9}};
    SMatrix sm2 =
        {1, 2, 3,
         4, 5, 6,
         7, 8, 9};
    Log::Info(LOG_NAME, "The sm is:");
    sm.PrintMatrix();
        Log::Info(LOG_NAME, "The sm2 is:");
    sm2.PrintMatrix();

    auto prv1 = sm.Get(1)->Data();

    Log::Debug(LOG_NAME, "prv1");

    prv1.PrintVector();

    Log::Debug(LOG_NAME, "sm * prv1");
    sm * prv1;
    prv1.PrintVector();

    auto sm_t = +sm;
    sm_t->Data().PrintMatrix();

    Log::Debug(LOG_NAME, "%d", &sm);

    SMatrix sm3 =
        {0, 3, 1,
         1, 5, 7,
         0, 2, 4};

    // Log::Debug(LOG_NAME, "%f", sm3.AComplement(1, 0)->Data());

    Log::Debug(LOG_NAME, "Adjoint matrix: ");
    auto asm3 = (*sm3)->Data();
    asm3.PrintMatrix();
    Log::Debug(LOG_NAME, "Inverse matrix: ");
    auto ism3 = (-sm3)->Data();
    ism3.PrintMatrix();
}

int main(int argc, char** argv)
{
    Log::Level(DEBUG_LEVEL);
    ///
    MatrixTest();
    ///
	system("pause");
	return 0;
}



