#pragma once
#include <vector>
#include "kamanri/utils/list.hpp"
#include "kamanri/utils/memory.hpp"
#include "kamanri/renderer/world/__/triangle3d.hpp"
#include "kamanri/maths/all.hpp"
namespace Kamanri
{
    namespace Renderer
    {
        namespace World
        {
            using RGB = unsigned long;
            namespace BlinnPhongReflectionModel$
            {
#ifdef __CUDA_RUNTIME_H__  
					__device__
#endif
				inline void AddHandle(unsigned int& y, RGB x) { y += x; }

#ifdef __CUDA_RUNTIME_H__  
					__device__
#endif
				inline void ReflectHandle(unsigned int& y, RGB x) { y *= x; y /= 0xFF; }

#ifdef __CUDA_RUNTIME_H__  
					__device__
#endif
                inline void DivideRGB(RGB color, unsigned int& r, unsigned int& g, unsigned int& b, void (*handle_func)(unsigned int&, RGB))
                {
                    handle_func(r, (color & 0x00ff0000) >> 16);
                    handle_func(g, (color & 0x0000ff00) >> 8);
                    handle_func(b, color & 0x000000ff);
                }

#ifdef __CUDA_RUNTIME_H__  
					__device__
#endif
                inline RGB CombineRGB(unsigned int r, unsigned int g, unsigned int b)
                {
                    return ((r < 0xff ? r : 0xff) << 16) |
                        ((g < 0xff ? g : 0xff) << 8) |
                        (b < 0xff ? b : 0xff);
                }

#ifdef __CUDA_RUNTIME_H__  
					__device__
#endif
                inline RGB RGBAdd(RGB c1, RGB c2)
                {
                    unsigned int r = 0, g = 0, b = 0;
                    DivideRGB(c1, r, g, b, AddHandle);
                    DivideRGB(c2, r, g, b, AddHandle);
                    return CombineRGB(r, g, b);
                }

#ifdef __CUDA_RUNTIME_H__  
					__device__
#endif
                inline RGB RGBAdd(RGB c1, RGB c2, RGB c3)
                {
                    unsigned int r = 0, g = 0, b = 0;
                    DivideRGB(c1, r, g, b, AddHandle);
                    DivideRGB(c2, r, g, b, AddHandle);
                    DivideRGB(c3, r, g, b, AddHandle);
                    return CombineRGB(r, g, b);
                }

#ifdef __CUDA_RUNTIME_H__  
					__device__
#endif
                inline RGB RGBReflect(RGB light, RGB reflect_point)
                {
                    unsigned int r = 0, g = 0, b = 0;
                    DivideRGB(light, r, g, b, AddHandle);
                    DivideRGB(reflect_point, r, g, b, ReflectHandle);
                    return CombineRGB(r, g, b);
                }

#ifdef __CUDA_RUNTIME_H__  
					__device__
#endif
                inline RGB RGBMul(RGB color, double num)
                {
                    unsigned int r = 0, g = 0, b = 0;
                    DivideRGB(color, r, g, b, AddHandle);
                    r = (unsigned int) (r * num);
                    g = (unsigned int) (g * num);
                    b = (unsigned int) (b * num);
                    return CombineRGB(r, g, b);
                }

                class PointLight
                {

                    public:
                    Maths::Vector location;
                    Maths::Vector location_model_view_transformed;
                    double power;
                    RGB color;

                    PointLight(Maths::Vector location, double power, DWORD color):
                        location(location), location_model_view_transformed(location), power(power), color(color)
                    {}

                };

                struct PointLightBufferItem
                {
                    bool is_specular = false;
                    bool is_exposed = true;
                    double specular_factor;
                    double distance = DBL_MAX;
                    PointLightBufferItem() = default;
                    PointLightBufferItem(bool is_exposed, double distance):
                        is_exposed(is_exposed), distance(distance)
                    {}
                };
            } // namespace BlinnPhongReflectionModel$

            namespace __
            {
                // declare a bounding box
                class BoundingBox;
            }

            class BlinnPhongReflectionModel
            {
                private:
                /* data */
                std::vector<BlinnPhongReflectionModel$::PointLight> _point_lights;
                Utils::List<BlinnPhongReflectionModel$::PointLight> _cuda_point_lights;

                // Note that its size is buffer_size * _point_lights.size()
                Utils::P<BlinnPhongReflectionModel$::PointLightBufferItem[]> _lights_buffer;
                BlinnPhongReflectionModel$::PointLightBufferItem* _cuda_lights_buffer;
                size_t _screen_width;
                size_t _screen_height;

                double _specular_min_cos;
                double _ambient_factor;
                double _diffuse_factor;

                bool _is_use_cuda;

#ifdef __CUDA_RUNTIME_H__  
                __device__
#endif
					void __BuildPerTriangleLightPixel(size_t x, size_t y, __::Triangle3D& triangle, size_t point_light_index, FrameBuffer& buffer);

                public:
                // BlinnPhongReflectionModel() = default;
                BlinnPhongReflectionModel(std::vector<BlinnPhongReflectionModel$::PointLight>&& point_lights, size_t screen_width, size_t screen_height, double specular_min_cos = 0.999, double diffuse_factor = 1 / (Maths::PI * 2), double ambient_factor = 0.1, bool is_use_cuda = false);
                BlinnPhongReflectionModel(BlinnPhongReflectionModel&& other);
                ~BlinnPhongReflectionModel();
                void DeleteCUDA();
                BlinnPhongReflectionModel& operator=(BlinnPhongReflectionModel&& other);
                void ModelViewTransform(Maths::SMatrix const& matrix);
                inline size_t ScreenWidth() { return _screen_width; }
                inline size_t ScreenHeight() { return _screen_height; }
#ifdef __CUDA_RUNTIME_H__  
                __device__
#endif
                    void InitLightBufferPixel(size_t x, size_t y, FrameBuffer& buffer);
#ifdef __CUDA_RUNTIME_H__  
                __device__
#endif
                    void __BuildPerTrianglePixel(size_t x, size_t y, __::Triangle3D& triangle, FrameBuffer& buffer);
#ifdef __CUDA_RUNTIME_H__  
                __device__
#endif
					void __BuildPixel(size_t x, size_t y, Utils::List<__::Triangle3D> triangles, __::BoundingBox* boxes, FrameBuffer& buffer);
#ifdef __CUDA_RUNTIME_H__  
                __device__
#endif
                    void WriteToPixel(size_t x, size_t y, FrameBuffer& buffer, DWORD& pixel);

            };


        } // namespace World

    } // namespace Renderer

} // namespace Kamanri
