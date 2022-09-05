#include <float.h>
#include "../../utils/logs.hpp"
#include "../../renderer/world3ds.hpp"
#include "../../renderer/triangle3ds.hpp"
#include "../../utils/result.hpp"
#include "../../maths/matrix.hpp"
#include "../../utils/memory.hpp"
#include "../../utils/string.hpp"

using namespace Kamanri::Utils::Logs;
using namespace Kamanri::Renderer::World3Ds;
using namespace Kamanri::Renderer::Triangle3Ds;
using namespace Kamanri::Maths::Vectors;
using namespace Kamanri::Maths::Matrix;
using namespace Kamanri::Utils::Result;
using namespace Kamanri::Utils::Memory;

constexpr const char* LOG_NAME = STR(Kamanri::Renderer::World3Ds);

World3D::World3D(Cameras::Camera& camera): _camera(camera)
{
    _camera.SetVertices(_vertices, _vertices_transform);
    
    _buffers.Init(_camera.ScreenWidth(), _camera.ScreenHeight());
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
        TRY(transform_matrix * (*_pvertices)[i]);
    }
    return DEFAULT_RESULT;
}

void Buffers::Init(unsigned int width, unsigned int height)
{
    _width = width;
    _height = height;
    z_buffer = P<double>(new double[width * height]);
    
}


void Buffers::WriteToZBufferFrom(Triangle3D const &t)
{
    int min_width;
    int min_height;
    int max_width;
    int max_height;

    t.GetMinMaxWidthHeight(min_width, min_height, max_width, max_height);

    auto p_z_buffer = z_buffer.get();
    double *p_z_buffer_i_j;
    double t_z;

    for (int i = min_width; i <= max_width; i++)
    {

        if (i >= _width || i < 0)
            continue;
        for (int j = min_height; j <= max_height; j++)
        {
            if (j >= _height || j < 0)
                continue;

            if(!t.IsIn(i, j)) 
                continue;

            // start to compare the depth
            p_z_buffer_i_j = p_z_buffer + i * _height + j;
            t_z = t.Z(i, j);
            if(t_z > *p_z_buffer_i_j)
                *p_z_buffer_i_j = t_z;
                
        }
    }
}
void Buffers::CleanZBuffer() const
{
    auto p_z_buffer = z_buffer.get();

    for(auto i = 0; i < _width; i++)
    {
        for(auto j = 0; j < _height; j++)
        {
            *(p_z_buffer + i * _height + j) = -DBL_MAX;
        }
    }
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
    _buffers.CleanZBuffer();
    for(auto i = _environment.triangles.begin(); i != _environment.triangles.end(); i++)
    {
        i->Build();
        i->PrintTriangle(is_print);
        _buffers.WriteToZBufferFrom(*i);
    }

    return DEFAULT_RESULT;
}

double World3D::Depth(int x, int y)
{
    
    x = x % _buffers.Height();
    y = y % _buffers.Width();
    
    return *(_buffers.z_buffer.get() + x * _buffers.Height() + y);
}

