#include <float.h>
#include "../../utils/logs.hpp"
#include "../../renderer/world3ds.hpp"
#include "../../renderer/triangle3ds.hpp"
#include "../../utils/result.hpp"

using namespace Kamanri::Utils::Logs;
using namespace Kamanri::Renderer::World3Ds;
using namespace Kamanri::Renderer::Triangle3Ds;
using namespace Kamanri::Maths::Vectors;
using namespace Kamanri::Utils::Result;

constexpr const char* LOG_NAME = "Kamanri::Renderer::World3D";

World3D::World3D(ObjReader::ObjModel const& model, Cameras::Camera& camera, bool is_print): _camera(camera)
{

    _camera.SetVertices(_vertices, _vertices_transform);

    auto dot_offset = (int)_vertices.size();

    for(auto i = 0; i < model.GetVerticeSize(); i++)
    {
        auto vertice = **model.GetVertice(i);
        Vector vector = {vertice[0], vertice[1], vertice[2], 1};
        _vertices.push_back(vector);
        _vertices_transform.push_back(vector);
    }


    for(auto i = 0; i < model.GetFaceSize(); i++)
    {
        auto face = **model.GetFace(i);
        if(face.vertice_indexes.size() > 4)
        {
            Log::Error(LOG_NAME, "Can not handle `face.vertice_indexes() > 4`");
            return;
        }
        if(face.vertice_indexes.size() == 4)
        {
            auto triangle = Triangle3D(_vertices_transform, dot_offset, face.vertice_indexes[0] - 1, face.vertice_indexes[3] - 1, face.vertice_indexes[2] - 1);
            this->_environment.triangles.push_back(triangle);
        }
        auto triangle2 = Triangle3D(_vertices_transform, dot_offset, face.vertice_indexes[0] - 1, face.vertice_indexes[1] - 1, face.vertice_indexes[2] - 1);
        
        _environment.triangles.push_back(triangle2);
    }

    // do check
    Log::Info(LOG_NAME, "Start to init World3D...");
    for(auto i = 0; i < _environment.triangles.size(); i++)
    {
        auto triangle = _environment.triangles[i];
        triangle.PrintTriangle(is_print);
    }
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
        x = **(*i)[0];
        y = **(*i)[1];
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