#include <float.h>
#include "../../utils/logs.hpp"
#include "../../renderer/world3ds.hpp"
#include "../../renderer/triangle3ds.hpp"
#include "../../utils/result.hpp"
#include "../../maths/matrix.hpp"
#include "../../utils/memory.hpp"

using namespace Kamanri::Utils::Logs;
using namespace Kamanri::Renderer::World3Ds;
using namespace Kamanri::Renderer::Triangle3Ds;
using namespace Kamanri::Maths::Vectors;
using namespace Kamanri::Maths::Matrix;
using namespace Kamanri::Utils::Result;
using namespace Kamanri::Utils::Memory;

constexpr const char* LOG_NAME = "Kamanri::Renderer::World3D";
SOURCE_FILE("kamanri/implements/renderer/world3d.cpp");

World3D::World3D(Cameras::Camera& camera): _camera(camera)
{
    _camera.SetVertices(_vertices, _vertices_transform);
}

Object::Object(std::vector<Maths::Vectors::Vector>& vertices, int offset, int length): _pvertices(&vertices), _offset(offset), _length(length){}

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
        TRY(transform_matrix * (*_pvertices)[i], 28);
    }
    return DEFAULT_RESULT;
}

PMyResult<Object> World3D::AddObjModel(ObjReader::ObjModel const &model, bool is_print)
{
    auto dot_offset = (int)_vertices.size();

    for(auto i = 0; i < model.GetVertexSize(); i++)
    {
        auto vertex = **model.GetVertex(i);
        Vector vector = {vertex[0], vertex[1], vertex[2], 1};
        _vertices.push_back(vector);
        _vertices_transform.push_back(vector);
    }


    for(auto i = 0; i < model.GetFaceSize(); i++)
    {
        auto face = **model.GetFace(i);
        if(face.vertex_indexes.size() > 4)
        {
            auto message = "Can not handle `face.vertex_indexes() > 4`";
            Log::Error(LOG_NAME, message);
            return RESULT_EXCEPTION(Object, WORLD3D_CODE_UNHANDLED_EXCEPTION, message);
        }
        if(face.vertex_indexes.size() == 4)
        {
            auto triangle = Triangle3D(_vertices_transform, dot_offset, face.vertex_indexes[0] - 1, face.vertex_indexes[3] - 1, face.vertex_indexes[2] - 1);
            this->_environment.triangles.push_back(triangle);
        }
        auto triangle2 = Triangle3D(_vertices_transform, dot_offset, face.vertex_indexes[0] - 1, face.vertex_indexes[1] - 1, face.vertex_indexes[2] - 1);
        
        _environment.triangles.push_back(triangle2);
    }

    // do check
    for(auto i = 0; i < _environment.triangles.size(); i++)
    {
        _environment.triangles[i].PrintTriangle(is_print);
    }

    Object result_object(_vertices, dot_offset, (int)model.GetVertexSize());
    return New<MyResult<Object>>(result_object);
}

DefaultResult World3D::Build(bool is_print)
{
    Log::Info(LOG_NAME, "Start to build the world...");
    for(auto i = _environment.triangles.begin(); i != _environment.triangles.end(); i++)
    {
        i->Build();
        i->PrintTriangle(is_print);
    }

    return DEFAULT_RESULT;
}

double World3D::Depth(double x, double y)
{
    double depth = -DBL_MAX;
    for(auto i = _environment.triangles.begin(); i != _environment.triangles.end(); i++)
    {
        if(!i->IsIn(x, y)) continue;
        auto z = i->Z(x, y);
        if(z <= depth) continue;
        depth = z;
    }

    return depth;
}

bool World3D::GetMinMaxWidthHeight(double &min_width, double &min_height, double &max_width, double& max_height)
{
    min_width = DBL_MAX;
    min_height = DBL_MAX;
    max_width = 0;
    max_height = 0;

    double x;
    double y;

    for (auto i = _vertices_transform.begin(); i != _vertices_transform.end(); i++)
    {
        x = i->GetFast(0);
        y = i->GetFast(1);
        if(x < min_width)
            min_width = x;
        if(x > max_width)
            max_width = x;
        if(y < min_height)
            min_height = y;
        if(y > max_height)
            max_height = y;
    }

    return true;
}