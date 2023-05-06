#include "kamanri/renderer/world/__/environment.hpp"
using namespace Kamanri::Renderer::World::__;

Environment& Environment::operator=(Environment const& other)
{
    bpr_model = other.bpr_model;
    triangles = other.triangles;
    cuda_triangles = other.cuda_triangles;

    objects = other.objects;
    cuda_objects = other.cuda_objects;
	
    for (size_t i = 0; i < objects.size(); i++)
    {   
        objects[i].__UpdateTriangleRef(triangles, objects, i);
    }
    return *this;
};

Environment& Environment::operator=(Environment&& other)
{
    bpr_model = std::move(other.bpr_model);
    triangles = std::move(other.triangles);
    cuda_triangles = other.cuda_triangles;

    objects = std::move(other.objects);
    cuda_objects = other.cuda_objects;
	
    for (size_t i = 0; i < objects.size(); i++)
    {   
        objects[i].__UpdateTriangleRef(triangles, objects, i);
    }
    return *this;
};