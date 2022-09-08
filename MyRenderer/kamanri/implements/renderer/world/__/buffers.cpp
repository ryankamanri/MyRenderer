#include <float.h>
#include "../../../../renderer/world/__/buffers.hpp"


using namespace Kamanri::Renderer::World::__;
using namespace Kamanri::Utils;

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