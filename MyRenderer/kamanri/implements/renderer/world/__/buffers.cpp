#include <cfloat>
#include "kamanri/utils/string.hpp"
#include "kamanri/renderer/world/__/buffers.hpp"


using namespace Kamanri::Renderer::World::__;
using namespace Kamanri::Utils;

namespace Kamanri
{
    namespace Renderer
    {
        namespace World
        {
            namespace __
            {
                namespace __Buffers
                {
                    constexpr const char* LOG_NAME = STR(Kamanri::Renderer::World::__::Buffers);
                } // namespace __Buffers
                
            } // namespace __
            
        } // namespace World
        
    } // namespace Renderer
    
} // namespace Kamanri


void Buffers::Init(unsigned int width, unsigned int height)
{
    _width = width;
    _height = height;
    _buffers = NewArray<FrameBuffer>(width * height);
    
}

Buffers& Buffers::operator=(Buffers&& other)
{
    _width = other._width;
    _height = other._height;
    _buffers = std::move(other._buffers);
    return *this;
}

void Buffers::CleanZBuffer() const
{
    for(auto i = 0; i < _width; i++)
    {
        for(auto j = 0; j < _height; j++)
        {
            _buffers[i * _height + j].z = -DBL_MAX;
        }
    }
}


void Buffers::WriteToZBufferFrom(Triangle3D const &t)
{
    int min_width;
    int min_height;
    int max_width;
    int max_height;

    t.GetMinMaxWidthHeight(min_width, min_height, max_width, max_height);

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
            t_z = t.Z(i, j);
            if(t_z > _buffers[i * _height + j].z)
                _buffers[i * _height + j].z = t_z;
                
        }
    }
}


FrameBuffer& Buffers::Get(int width, int height)
{
    if(width < 0 || height < 0 || width >= _width || height >= _height)
    {
        Log::Error(__Buffers::LOG_NAME, "Invalid Index (%d, %d), return the 0 index content", width, height);
        return _buffers[0];
    }
    return _buffers[width * _height + height];
    
}