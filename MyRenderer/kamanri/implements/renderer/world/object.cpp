#include "kamanri/utils/string.hpp"
#include "kamanri/renderer/world/object.hpp"
#include "kamanri/renderer/world/__/triangle3d.hpp"
#include "kamanri/utils/result.hpp"
#include "kamanri/maths/matrix.hpp"

using namespace Kamanri::Renderer::World;
using namespace Kamanri::Utils;
using namespace Kamanri::Maths;

namespace Kamanri
{
	namespace Renderer
	{
		namespace World
		{
			namespace __Object
			{
				constexpr const char* LOG_NAME = STR(Kamanri::Renderer::World::Object);
			} // namespace __Object
			
		} // namespace World
		
	} // namespace Renderer
	
} // namespace Kamanri


Object::Object(std::vector<Maths::Vector>& vertices, size_t v_offset, size_t v_length, size_t t_offset, size_t t_length, std::string tga_image_name): 
_pvertices(&vertices), _v_offset(v_offset), _v_length(v_length), _t_offset(t_offset), _t_length(t_length)
{
	if(!_img.ReadTGAFile(tga_image_name))
	{
		Log::Error(__Object::LOG_NAME, "Cannot read the TGA image '%s'.", tga_image_name.c_str());
		PRINT_LOCATION;
	}
}

void Object::__UpdateTriangleRef(std::vector<__::Triangle3D>& triangles, std::vector<Object>& objects, size_t index)
{
	for(size_t i = _t_offset; i < _t_offset + _t_length; i++)
	{
		auto& t = triangles[i];
		t._p_objects = &objects;
		t._index = index;
	}
}

DefaultResult Object::Transform(SMatrix const& transform_matrix) const
{
	for(size_t i = _v_offset; i < _v_offset + _v_length; i++)
	{
		transform_matrix * (*_pvertices)[i];
	}
	return DEFAULT_RESULT;
}