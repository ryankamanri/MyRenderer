#include "kamanri/renderer/world/world3d.hpp"
#include "kamanri/utils/string.hpp"
#include "cuda_dll/exports/build_world.hpp"

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
				
				namespace Build
				{
					dll cuda_dll;
					func_type(BuildWorld) build_world;
				} // namespace Build

				void Import_BuildWorld()
				{
					load_dll(cuda_dll, Build::cuda_dll, LOG_NAME);
					import_func(BuildWorld, Build::cuda_dll, Build::build_world, LOG_NAME);
				}
				
			} // namespace __World3D
			
		} // namespace World
		
	} // namespace Renderer
	
} // namespace Kamanri



World3D::World3D(): _camera(Camera()) 
{
	__World3D::Import_BuildWorld();
}

World3D::World3D(Camera&& camera): _camera(std::move(camera))
{
	_camera.SetVertices(_resources.vertices, _resources.vertices_transformed, _resources.vertices_model_view_transformed);
	
	_buffers.Init(_camera.ScreenWidth(), _camera.ScreenHeight());

	__World3D::Import_BuildWorld();
}

World3D& World3D::operator=(World3D && other)
{
	_resources.vertices = std::move(other._resources.vertices);
	_resources.vertices_transformed = std::move(other._resources.vertices_transformed);
	_resources.vertices_model_view_transformed = std::move(other._resources.vertices_model_view_transformed);
	_resources.vertex_textures = std::move(other._resources.vertex_textures);
	_resources.vertex_normals = std::move(other._resources.vertex_normals);
	_camera = std::move(other._camera);
	_environment = std::move(other._environment);
	_buffers = std::move(other._buffers);
	// Move the reference of vertices of camera
	_camera.SetVertices(_resources.vertices, _resources.vertices_transformed, _resources.vertices_model_view_transformed);
	return *this;
}

Result<Object *> World3D::AddObjModel(ObjModel const &model)
{
	auto v_offset = (int)_resources.vertices.size();
	auto vt_offset = (int)_resources.vertex_textures.size();
	auto vn_offset = (int)_resources.vertex_normals.size();

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
		Vector vector = {vertex[0], vertex[1], 1};
		_resources.vertex_normals.push_back(vector);
	}

	for(size_t i = 0; i < model.GetVertexTextureSize(); i++)
	{
		auto vertex = *model.GetVertexTexture(i);
		Vector vector = {vertex[0], vertex[1], vertex[2], 1};
		_resources.vertex_textures.push_back(vector);
	}

	// Add an object
	_environment.objects.push_back(Object(_resources.vertices, v_offset, (int)model.GetVertexSize(), model.GetTGAImageName()));
	// Now you can get the object& by _environment.objects.back()
	auto& object = _environment.objects.back();


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
				object,
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
			object,
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

	// do check
	for(size_t i = 0; i < _environment.triangles.size(); i++)
	{
		_environment.triangles[i].PrintTriangle(Log$::TRACE_LEVEL);
	}

	return Result<Object *>(&object);
}


World3D&& World3D::AddObjModel(ObjModel const& model, Maths::SMatrix const& transform_matrix)
{
	auto res = AddObjModel(model);
	if(res.IsException())
	{
		res.Print();
	}
	res.Data()->Transform(transform_matrix);
	return std::move(*this);
}



DefaultResult World3D::Build()
{
	Log::Debug(__World3D::LOG_NAME, "Start to build the world...");
	// TODO: CUDA parallelize "Build World"

	_buffers.CleanAllBuffers();
	for(auto& t: _environment.triangles)
	{
		t.Build(_resources);
		t.PrintTriangle(Log$::TRACE_LEVEL);
		t.WriteTo(_buffers, _camera.NearestDist());

	}
	
	//
	return DEFAULT_RESULT;
}

FrameBuffer const& World3D::FrameBuffer(int x, int y)
{
	
	x = x % _buffers.Height();
	y = y % _buffers.Width();
	
	return _buffers.GetFrame(x, y);
}
