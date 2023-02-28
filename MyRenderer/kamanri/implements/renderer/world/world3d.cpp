#include "kamanri/renderer/world/world3d.hpp"
#include "kamanri/utils/string.hpp"
#include "cuda_dll/exports/build_world.hpp"
#include "cuda_dll/exports/memory_operations.hpp"
#include "cuda_dll/exports/write_to_buffers.hpp"

using namespace Kamanri::Renderer::World;
using namespace Kamanri::Maths;
using namespace Kamanri::Utils;

namespace Kamanri
{
	namespace Renderer
	{
		namespace World
		{
			namespace __World3D
			{
				constexpr const char* LOG_NAME = STR(Kamanri::Renderer::World::World3D);

				dll cuda_dll;
				func_type(CUDAMalloc) cuda_malloc;
				func_type(CUDAFree) cuda_free;
				func_type(TransmitToCUDA) transmit_to_cuda;
				func_type(TransmitFromCUDA) transmit_from_cuda;

				namespace Build
				{
					func_type(BuildWorld) build_world;
					func_type(WriteToBuffers) write_to_buffers;
				} // namespace Build

				void ImportFunctions()
				{
					load_dll(cuda_dll, cuda_dll, LOG_NAME);
					import_func(CUDAMalloc, cuda_dll, cuda_malloc, LOG_NAME);
					import_func(CUDAFree, cuda_dll, cuda_free, LOG_NAME);
					import_func(TransmitToCUDA, cuda_dll, transmit_to_cuda, LOG_NAME);
					import_func(TransmitFromCUDA, cuda_dll, transmit_from_cuda, LOG_NAME);
					import_func(BuildWorld, cuda_dll, Build::build_world, LOG_NAME);
					import_func(WriteToBuffers, cuda_dll, Build::write_to_buffers, LOG_NAME);
				}
				
			} // namespace __World3D
			
		} // namespace World
		
	} // namespace Renderer
	
} // namespace Kamanri




World3D::World3D(Camera&& camera, BlingPhongReflectionModel&& model)
: _camera(std::move(camera)), 
_buffers(_camera.ScreenWidth(), _camera.ScreenHeight()),
_environment(std::move(model))
{
	_camera.__SetRefs(_resources, _environment.bpr_model);
	__World3D::ImportFunctions();
}



World3D& World3D::operator=(World3D && other)
{
	_resources = std::move(other._resources);
	_camera = std::move(other._camera);
	_environment = std::move(other._environment);
	_buffers = std::move(other._buffers);
	_configs = std::move(other._configs);
	// Move the reference of vertices of camera
	_camera.__SetRefs(_resources, _environment.bpr_model);
	return *this;
}

Result<Object *> World3D::AddObjModel(ObjModel const &model)
{
	auto v_offset = _resources.vertices.size();
	auto vt_offset = _resources.vertex_textures.size();
	auto vn_offset = _resources.vertex_normals.size();

	/// transform to Vector from std::vector

	for(size_t i = 0; i < model.GetVertexSize(); i++)
	{
		auto vertex = *model.GetVertex(i);
		Vector vector = {vertex[0], vertex[1], vertex[2], 1};
		_resources.vertices.push_back(vector);
		_resources.vertices_transformed.push_back(vector);
		_resources.vertices_model_view_transformed.push_back(vector);
		
	}

	for(size_t i = 0; i < model.GetVertexNormalSize(); i++)
	{
		auto vertex = *model.GetVertexNormal(i);
		Vector vector = {vertex[0], vertex[1], vertex[2], 0};
		_resources.vertex_normals.push_back(vector);
		_resources.vertex_normals_model_view_transformed.push_back(vector);
	}

	for(size_t i = 0; i < model.GetVertexTextureSize(); i++)
	{
		auto vertex = *model.GetVertexTexture(i);
		Vector vector = {vertex[0], vertex[1], vertex.size() > 2 ? vertex[2] : 0};
		_resources.vertex_textures.push_back(vector);
	}

	auto t_offset = _environment.triangles.size();

	for(size_t i = 0; i < model.GetFaceSize(); i++)
	{
		auto face = *model.GetFace(i);
		if(face.vertex_indexes.size() > 4)
		{
			auto message = "Can not handle `face.vertex_indexes() > 4`";
			Log::Error(__World3D::LOG_NAME, message);
			PRINT_LOCATION;
			return RESULT_EXCEPTION(Object *, World3D$::CODE_UNHANDLED_EXCEPTION, message);
		}
		// Some object may not have vns
		auto has_vn = face.vertex_normal_indexes.size() != 0;
		if (face.vertex_indexes.size() == 4)
		{
			auto splited_triangle = __::Triangle3D(
				_environment.objects,
				_environment.objects.size(),
				_environment.triangles.size(),
				v_offset + face.vertex_indexes[0] - 1,
				v_offset + face.vertex_indexes[3] - 1,
				v_offset + face.vertex_indexes[2] - 1,
				vt_offset + face.vertex_texture_indexes[0] - 1,
				vt_offset + face.vertex_texture_indexes[3] - 1,
				vt_offset + face.vertex_texture_indexes[2] - 1,
				has_vn ? vn_offset + face.vertex_normal_indexes[0] - 1 : __::Triangle3D$::INEXIST_INDEX,
				has_vn ? vn_offset + face.vertex_normal_indexes[3] - 1 : __::Triangle3D$::INEXIST_INDEX,
				has_vn ? vn_offset + face.vertex_normal_indexes[2] - 1 : __::Triangle3D$::INEXIST_INDEX);
			this->_environment.triangles.push_back(splited_triangle);
		}
		auto triangle = __::Triangle3D(
			_environment.objects,
			_environment.objects.size(),
			_environment.triangles.size(),
			v_offset + face.vertex_indexes[0] - 1,
			v_offset + face.vertex_indexes[1] - 1,
			v_offset + face.vertex_indexes[2] - 1,
			vt_offset + face.vertex_texture_indexes[0] - 1,
			vt_offset + face.vertex_texture_indexes[1] - 1,
			vt_offset + face.vertex_texture_indexes[2] - 1,
			has_vn ? vn_offset + face.vertex_normal_indexes[0] - 1 : __::Triangle3D$::INEXIST_INDEX,
			has_vn ? vn_offset + face.vertex_normal_indexes[1] - 1 : __::Triangle3D$::INEXIST_INDEX,
			has_vn ? vn_offset + face.vertex_normal_indexes[2] - 1 : __::Triangle3D$::INEXIST_INDEX);

		_environment.triangles.push_back(triangle);
	}

	// Add an object
	_environment.objects.push_back(Object(_resources.vertices, v_offset, model.GetVertexSize(), t_offset, model.GetFaceSize(), model.GetTGAImageName()));
	// Now you can get the object& by _environment.objects.back()
	auto& object = _environment.objects.back();

	return Result<Object *>(&object);
}


World3D& World3D::AddObjModel(ObjModel const& model, Maths::SMatrix const& transform_matrix)
{
	auto res = AddObjModel(model);
	if(res.IsException())
	{
		res.Print();
	}
	res.Data()->Transform(transform_matrix);
	return *this;
}

World3D::~World3D()
{
	// __World3D::cuda_free(_environment.cuda_triangles);
	// __World3D::cuda_free(_environment.cuda_objects);
}

World3D& World3D::Commit(bool is_use_cuda)
{
	_configs.is_commited = true;

	_configs.is_use_cuda = is_use_cuda;
	if(!_configs.is_use_cuda) return *this;
	
	// objects
	auto objects_size = _environment.objects.size() * sizeof(Object);
	__World3D::cuda_malloc(&(void*)_environment.cuda_objects, objects_size);
	__World3D::transmit_to_cuda(&_environment.objects[0], _environment.cuda_objects, objects_size);

	return *this;
}

DefaultResult World3D::Build()
{
	if(!_configs.is_commited)
	{
		Log::Warn(__World3D::LOG_NAME, "World3D not commited, build may not avaliable");
	}

	Log::Debug(__World3D::LOG_NAME, "Start to build the world...");
	// TODO: CUDA parallelize "Build World"

	_buffers.CleanBitmap();

	for(auto& t: _environment.triangles)
	{
		t.Build(_resources);
	}

	for(size_t x = 0; x < _buffers.Width(); x++)
	{
		for(size_t y = 0; y < _buffers.Height(); y++)
		{
			BuildForPixel(x, y);
		}
	}
	///////////////////
	// triangles transmission
	// auto triangles_size = _environment.triangles.size() * sizeof(__::Triangle3D);
	// __World3D::transmit_to_cuda(&_environment.triangles[0], _environment.cuda_triangles, triangles_size);
	// __World3D::Build::write_to_buffers(
	// 	_environment.cuda_triangles, 
	// 	_environment.triangles.size(),
	// 	_buffers.CUDAGetBuffersPtr(), 
	// 	_buffers.CUDAGetBitmapBufferPtr(), 
	// 	_buffers.Width(), 
	// 	_buffers.Height(), 
	// 	_camera.NearestDist());
	// Log::Debug(__World3D::LOG_NAME, "cuda triangles at %p,", _environment.cuda_triangles);
	
	////////////////////
	
	//
	return DEFAULT_RESULT;
}

void World3D::BuildForPixel(size_t x, size_t y)
{
	// set z = infinity
	_buffers.InitPixel(x, y);

	auto& buffer = _buffers.GetFrame(x, y);
	auto& bitmap_pixel = _buffers.GetBitmapBuffer(x, y);

	for(auto& t: _environment.triangles)
	{
		t.WriteToPixel(x, y, buffer, bitmap_pixel, _camera.NearestDist());
	}

	if(_buffers.GetFrame(x, y).location[2] == -DBL_MAX) return;

	// set distance = infinity, is exposed.
	_environment.bpr_model.InitLightBufferPixel(x, y, buffer);
	for(auto& t: _environment.triangles)
	{
		_environment.bpr_model.__BuildPerTrianglePixel(x, y, t, buffer);
	}

	_environment.bpr_model.WriteToPixel(x, y, buffer, bitmap_pixel);
	
}

FrameBuffer const& World3D::FrameBuffer(int x, int y)
{
	
	x = x % _buffers.Width();
	y = y % _buffers.Height();
	
	return _buffers.GetFrame(x, y);
}
