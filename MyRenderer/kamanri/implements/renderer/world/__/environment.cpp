#include "kamanri/renderer/world/__/environment.hpp"
using namespace Kamanri::Renderer::World::__;

Environment& Environment::operator=(Environment&& other)
{
    triangles = std::move(other.triangles);
    cuda_triangles = other.cuda_triangles;
    objects = std::move(other.objects);
    for (size_t i = 0; i < objects.size(); i++)
    {   
        objects[i].__UpdateTriangleRef(triangles, objects, i);
    }
    return *this;
};