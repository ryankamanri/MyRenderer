// CMakeProject1.cpp: 定义应用程序的入口点。
//
#include <iostream>
#include "mylibs/mylog.h"
#include "renderers/tgaimage.h"
#include "renderers/model.h"
#include "renderers/line.h"
#include "mylibs/myresult.h"

constexpr int width  = 800; // output image size
constexpr int height = 800;

const TGAColor white = TGAColor(255, 255, 255, 255);
const TGAColor red   = TGAColor(255, 0,   0,   255);
const TGAColor green = TGAColor(0, 255, 0, 255);
Model *model = NULL;

SOURCE_FILE("../Main.cpp");


///////////////// make_unique



void Triangle(Vec2i t0, Vec2i t1, Vec2i t2, TGAImage &image, TGAColor color) { 
    Line(t0, t1, image, color); 
    Line(t1, t2, image, color); 
    Line(t2, t0, image, color); 
}


P<MyResult<int>> ResultTest() {
    auto innerResult = New<MyResult<int>>(MyResult<int>::EXCEPTION, 500, "Invalid Inner Result", 1);
    innerResult->PushToStack(StackTrace(SOURCE_FILE_LOCATION, 33));

    auto result = New<MyResult<int>>(MyResult<int>::EXCEPTION, 500, "Invalid Arguments", 9090, std::move(innerResult));
    result->PushToStack(StackTrace(SOURCE_FILE_LOCATION, 35));
    return result;
}

P<MyResult<const char*>> ResultTest2() {
    auto result1 = ResultTest();

    THROW_EXCEPTION(result1, 41, "const char*");
    
    auto data = result1->Data();
    Log::Info("name", "receive result1 data: %d", data);
    return New<MyResult<const char*>>("const char*");
}


int main(int argc, char** argv)
{
    Log::Level(INFO_LEVEL);
	Log::Info("TGAImage", "This is my TinyRenderer");
    PrintLn("Let's look about it");
	///
    // Vec2i t0[3] = {Vec2i(10, 70),   Vec2i(50, 160),  Vec2i(70, 80)}; 
    // Vec2i t1[3] = {Vec2i(180, 50),  Vec2i(150, 1),   Vec2i(70, 180)}; 
    // Vec2i t2[3] = {Vec2i(180, 150), Vec2i(120, 160), Vec2i(130, 180)}; 
    // TGAImage image(800, 800, TGAImage::RGB);
    // Triangle(t0[0], t0[1], t0[2], image, red); 
    // Triangle(t1[0], t1[1], t1[2], image, white); 
    // Triangle(t2[0], t2[1], t2[2], image, green);

    // const char* outPath = "out/out_image_2.tga";
    // std::string a = "12";
    // a += "3";
    // Log<>::Info("a", a.c_str());

    // image.write_tga_file(outPath);
    // Log<const char*>::Info( "Main", "Save File To %s", outPath);


    auto result = ResultTest2();


    result->PushToStack(StackTrace(SOURCE_FILE_LOCATION, 76));
    
    if(result->IsException()) 
    {
        if(result->Code() == 500)
        {
            Log::Info("name", "result is exception ? %d", result->IsException());
            Log::Info("name", "result is exception ? %d", result->IsException());
        }

        result->Print();

        Log::Info("name", "Shut down.");
        
        system("pause");
        return -1;
    }
        
    const char* data = result->Data();
    Log::Info("name", "receive data %s", data);

    ///
	system("pause");
	return 0;
}



