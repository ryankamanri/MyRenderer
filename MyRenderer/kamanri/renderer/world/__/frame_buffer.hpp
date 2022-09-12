#pragma once

namespace Kamanri
{
    namespace Renderer
    {
        namespace World
        {
            namespace __
            {
                class FrameBuffer
                {
                public:
                    enum class Type
                    {
                        Z, COLOR, INTENSITY, ALL
                    };
                    double z;
                    // BGR color
                    unsigned int color;
                    double intensity;
                };

            } // namespace __

        } // namespace World

    } // namespace Renderer

} // namespace Kamanri