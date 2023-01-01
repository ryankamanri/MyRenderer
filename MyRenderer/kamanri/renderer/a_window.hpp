// TODO: abstract window class
#pragma once
#include "kamanri/utils/delegate.hpp"
#include "kamanri/renderer/world/world3d.hpp"
namespace Kamanri
{
    namespace WindowProcedures
    {
        namespace Window$
        {
            using Color = unsigned long;
            class Painter;
            class PainterFactor
            {
            public:
                virtual Painter CreatePainter();
                virtual void Clean(Painter &painter);
            };
            class Painter
            {
            public:
                virtual void Flush() const;
                virtual inline void Dot(int x, int y, Color color);
                friend void PainterFactor::Clean(Painter &painter);
            };

        } // namespace Window$

        class Window
        {
        public:
            virtual Window& SetWorld(Renderer::World::World3D world);
            virtual Window &AddProcedure(Utils::Delegate$::Node proc);
            virtual Window& Show();
            virtual Window& Update();
            virtual void MessageLoop();
        };
    } // namespace WindowProcedures

} // namespace Kamanri
