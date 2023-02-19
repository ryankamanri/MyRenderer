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


Object::Object(std::vector<Maths::Vector>& vertices, int offset, int length, std::string tga_image_name): _pvertices(&vertices), _offset(offset), _length(length)
{
	if(!_img.ReadTGAFile(tga_image_name))
	{
		Log::Error(__Object::LOG_NAME, "Cannot read the TGA image '%s'.", tga_image_name.c_str());
		PRINT_LOCATION;
	}
}

// Object::Object(Object const& obj): _pvertices(obj._pvertices), _offset(obj._offset), _length(obj._length), _img(obj._img)
// {

// }

// Object& Object::operator=(Object& obj)
// {
//     _pvertices = obj._pvertices;
//     _offset = obj._offset;
//     _length = obj._length;
//     _img = obj._img;
//     return *this;
// }

void Object::__UpdateTriangleRef(std::vector<__::Triangle3D>& triangles)
{
	for(int i = _offset; i < _offset + _length; i++)
	{
		auto& t = triangles[i];
		t._p_object = this;
	}
}

DefaultResult Object::Transform(SMatrix const& transform_matrix) const
{
	for(int i = _offset; i < _offset + _length; i++)
	{
		transform_matrix * (*_pvertices)[i];
	}
	return DEFAULT_RESULT;
}