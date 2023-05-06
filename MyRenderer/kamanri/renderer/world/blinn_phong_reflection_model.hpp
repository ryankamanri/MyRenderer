#pragma once
#ifndef SWIG
#include "kamanri/renderer/world/blinn_phong_reflection_model$.hpp"
#endif
namespace Kamanri
{
    namespace Renderer
    {
        namespace World
        {

            namespace __
            {
                // declare a bounding box
                class BoundingBox;
            }

            class BlinnPhongReflectionModel
            {
                private:
                /* data */
                std::vector<Kamanri::Renderer::World::BlinnPhongReflectionModel$::PointLight> _point_lights;
                Kamanri::Utils::List<Kamanri::Renderer::World::BlinnPhongReflectionModel$::PointLight> _cuda_point_lights;

                // Note that its size is buffer_size * _point_lights.size()
                Kamanri::Utils::P<Kamanri::Renderer::World::BlinnPhongReflectionModel$::PointLightBufferItem[]> _lights_buffer;
                Kamanri::Renderer::World::BlinnPhongReflectionModel$::PointLightBufferItem* _cuda_lights_buffer;
                size_t _screen_width;
                size_t _screen_height;

                double _specular_min_cos;
                double _ambient_factor;
                double _diffuse_factor;

                bool _is_use_cuda;

#ifdef __CUDA_RUNTIME_H__  
                __device__
#endif
					void __BuildPerTriangleLightPixel(size_t x, size_t y, Kamanri::Renderer::World::__::Triangle3D& triangle, size_t point_light_index, Kamanri::Renderer::World::FrameBuffer& buffer);

                public:
                // BlinnPhongReflectionModel() = default;
                BlinnPhongReflectionModel(std::vector<Kamanri::Renderer::World::BlinnPhongReflectionModel$::PointLight>&& point_lights, size_t screen_width, size_t screen_height, double specular_min_cos = 0.999, double diffuse_factor = 1 / (Maths::PI * 2), double ambient_factor = 0.1, bool is_use_cuda = false);
                BlinnPhongReflectionModel(BlinnPhongReflectionModel&& other);
                ~BlinnPhongReflectionModel();
                void DeleteCUDA();
                BlinnPhongReflectionModel& operator=(BlinnPhongReflectionModel const& other);
                BlinnPhongReflectionModel& operator=(BlinnPhongReflectionModel&& other);
                void ModelViewTransform(Kamanri::Maths::SMatrix const& matrix);
                inline size_t ScreenWidth() { return _screen_width; }
                inline size_t ScreenHeight() { return _screen_height; }
#ifdef __CUDA_RUNTIME_H__  
                __device__
#endif
                    void InitLightBufferPixel(size_t x, size_t y, Kamanri::Renderer::World::FrameBuffer& buffer);
#ifdef __CUDA_RUNTIME_H__  
                __device__
#endif
                    void __BuildPerTrianglePixel(size_t x, size_t y, Kamanri::Renderer::World::__::Triangle3D& triangle, Kamanri::Renderer::World::FrameBuffer& buffer);
#ifdef __CUDA_RUNTIME_H__  
                __device__
#endif
					void __BuildPixel(size_t x, size_t y, Kamanri::Utils::List<Kamanri::Renderer::World::__::Triangle3D> triangles, Kamanri::Renderer::World::__::BoundingBox* boxes, Kamanri::Renderer::World::FrameBuffer& buffer);
#ifdef __CUDA_RUNTIME_H__  
                __device__
#endif
                    void WriteToPixel(size_t x, size_t y, Kamanri::Renderer::World::FrameBuffer& buffer, Kamanri::Renderer::World::RGB& pixel);

            };


        } // namespace World

    } // namespace Renderer

} // namespace Kamanri
