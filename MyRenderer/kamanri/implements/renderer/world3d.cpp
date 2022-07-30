#include "../../renderer/world3ds.h"


using namespace Kamanri::Renderer::World3Ds;
using namespace Kamanri::Maths::Vectors;

World3D::World3D(ObjReader::ObjModel const& model, Cameras::Camera& camera)
{
    _camera = camera;

    for(auto i = 0; i < model.GetVerticeSize(); i++)
    {
        auto vertice = **model.GetVertice(i);
        Vector vector = {vertice[0], vertice[1], vertice[2], vertice[3]};
        // _space_dots.push_back(vector);
    }

}

