#include <iostream>
#include <thread>
#include <cfloat>
#include <cmath>
#include "kamanri/all.hpp"
using namespace Kamanri::Maths;
using namespace Kamanri::Windows;
using namespace Kamanri::Windows::WinGDI_Window$;
using namespace Kamanri::WindowProcedures::WinGDI_Window;
using namespace Kamanri::Renderer;
using namespace Kamanri::Renderer::World;
using namespace Kamanri::Utils;


constexpr const char* LOG_NAME = "Main";
constexpr const int WINDOW_LENGTH = 800;
constexpr const bool IS_SHADOW_MAPPING = true;
constexpr const bool IS_OFFLINE = false;
constexpr const bool IS_USE_CUDA = true;
constexpr const unsigned int FRAME_COUNT = 128;


#define MODEL_DIABLO3_POSE
#define MODEL_SHIBA
#define MODEL_STRAWBERRY



#define BASE_PATH "C:/Users/97448/totFolder/source/repos/MyRenderer/MyRenderer/models/"

constexpr const char* DIABLO3_POSE_OBJ = BASE_PATH "diablo3_pose/diablo3_pose.obj";
constexpr const char* DIABLO3_POSE_TGA = BASE_PATH "diablo3_pose/diablo3_pose_diffuse.tga";
constexpr const char* SHIBA_OBJ = BASE_PATH "shiba/Shiba_Obj/Shiba.obj";
constexpr const char* SHIBA_TGA = BASE_PATH "shiba/Textures/Shiba_DIF01.tga";
constexpr const char* FLOOR_OBJ = BASE_PATH "floor/floor.obj";
constexpr const char* FLOOR_TGA = BASE_PATH "floor/floor_diffuse.tga";
constexpr const char* STRAWBERRY_OBJ = BASE_PATH "strawberry/Strawberry_obj.obj";
constexpr const char* STRAWBERRY_TGA = BASE_PATH "strawberry/Texture/Strawberry_basecolor.tga";

SMatrix DIABLO3_POSE_TRANSFORMER =
{
	2, 0, 0, 0,
	0, 2, 0, 0,
	0, 0, 2, 1,
	0, 0, 0, 1
};

SMatrix SHIBA_TRANSFORMER = 
{
	4, 0, 0, 0,
	0, 4, 0, -2,
	0, 0, 4, 1,
	0, 0, 0, 1
};

SMatrix STRAWBERRY_TRANSFORMER = 
{
	0.2, 0, 0, 1,
	0, 0.2, 0, -2,
	0, 0, 0.2, 1.5,
	0, 0, 0, 1
};




namespace __UpdateFunc
{
	Vector direction(4);
	double theta = PI / (FRAME_COUNT / 2);
	SMatrix revolve_matrix =
	{
		cos(theta), 0, -sin(theta), 0,
		0, 1, 0, 0,
		sin(theta), 0, cos(theta), 0,
		0, 0, 0, 1
	};

	bool is_transformed_once = false;
} // namespace __UpdateFunc

int UpdateFunc(World3D& world)
{
	using namespace __UpdateFunc;
	Camera& camera = world.GetCamera();
	direction = camera.Direction();

	camera.Transform(!is_transformed_once);
	is_transformed_once = true;

	world.Build();

	revolve_matrix* camera.Direction();
	revolve_matrix* camera.Location();

	camera.InverseUpperByDirection(direction);

	return 0;
}

void StartRender(HINSTANCE hInstance)
{
	World3D world(
		Camera(
			{ 0, -1, 5, 1 },
			{ 0, 0, -1, 0 },
			{ 0, 1, 0, 0 },
			-1,
			-5,
			WINDOW_LENGTH,
			WINDOW_LENGTH
		),
		BlinnPhongReflectionModel({
			BlinnPhongReflectionModel$::PointLight({2, 3, 4, 1}, 800, 0xffffff)
		}, WINDOW_LENGTH, WINDOW_LENGTH, 0.95, 1 / PI * 2, 0.4, IS_USE_CUDA),
		IS_SHADOW_MAPPING, IS_USE_CUDA
	);


	world
#ifdef MODEL_DIABLO3_POSE
	.AddObjModel(
		ObjModel(DIABLO3_POSE_OBJ, DIABLO3_POSE_TGA),
		DIABLO3_POSE_TRANSFORMER
	)
#endif
#ifdef MODEL_SHIBA
	.AddObjModel(
		ObjModel(SHIBA_OBJ, SHIBA_TGA),
		SHIBA_TRANSFORMER
	)
#endif
#ifdef MODEL_STRAWBERRY
	.AddObjModel(
		ObjModel(STRAWBERRY_OBJ, STRAWBERRY_TGA),
		STRAWBERRY_TRANSFORMER
	)
#endif
	.AddObjModel(
		ObjModel(FLOOR_OBJ, FLOOR_TGA),
		{
			2, 0, 0, 0,
			0, 2, 0, 0,
			0, 0, 2, 0,
			0, 0, 0, 1
		}
	).Commit();
	WinGDI_Window(hInstance, world, WINDOW_LENGTH, WINDOW_LENGTH)
		//.AddProcedure(MovePositionProcedure(0.05, PI / 256))
		.AddProcedure(
			UpdateProcedure(UpdateFunc, WINDOW_LENGTH, WINDOW_LENGTH, IS_OFFLINE, FRAME_COUNT, 100)
		).Show().MessageLoop();
}



//////////////////////////////////////////////////////

int main()

{
	HINSTANCE instance = GetModuleHandle("MyRenderer.exe");
	// set the log level and is_print
	Log::SetLevel(Log$::DEBUG_LEVEL);
	Log::Info(LOG_NAME, "%p: May you have a nice day!", instance);

	StartRender(instance);

	return 0;
}
