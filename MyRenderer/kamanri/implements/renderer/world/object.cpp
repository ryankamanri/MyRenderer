#include "kamanri/renderer/world/object.hpp"
#include "kamanri/utils/result.hpp"
#include "kamanri/maths/matrix.hpp"

using namespace Kamanri::Renderer::World;
using namespace Kamanri::Utils;
using namespace Kamanri::Maths;

Object::Object(std::vector<Maths::Vector>& vertices, int offset, int length): _pvertices(&vertices), _offset(offset), _length(length){}

Object::Object(Object& obj): _pvertices(obj._pvertices), _offset(obj._offset), _length(obj._length)
{

}

Object& Object::operator=(Object& obj)
{
    _pvertices = obj._pvertices;
    _offset = obj._offset;
    _length = obj._length;
    return *this;
}

DefaultResult Object::Transform(SMatrix const& transform_matrix) const
{
    for(int i = _offset; i < _offset + _length; i++)
    {
        ASSERT(transform_matrix * (*_pvertices)[i]);
    }
    return DEFAULT_RESULT;
}