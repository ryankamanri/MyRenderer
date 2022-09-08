#pragma once
#include "../../../utils/memory.hpp"
#include "triangle3d.hpp"

namespace Kamanri
{
    namespace Renderer
    {
        namespace World
        {
            namespace __
            {
                class Buffers
                {
                private:
                    unsigned int _width;
                    unsigned int _height;

                public:
                    Utils::P<double> z_buffer;
                    void Init(unsigned int width, unsigned int height);
                    void CleanZBuffer() const;
                    void WriteToZBufferFrom(Triangle3D const &t);
                    inline int Width() const { return _width; }
                    inline int Height() const { return _height; }
                };

            } // namespace __

        } // namespace World

    } // namespace Renderer

} // namespace Kamanri