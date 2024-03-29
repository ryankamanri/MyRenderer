#pragma once
#ifndef SWIG
#include <vector>
#include "kamanri/utils/list.hpp"
#include "kamanri/utils/memory.hpp"
#include "kamanri/renderer/world/__/triangle3d.hpp"
#include "kamanri/maths/all.hpp"
#endif
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
                    Kamanri::Maths::Vector location;
                    Kamanri::Maths::Vector location_model_view_transformed;
                    double power;
                    RGB color;

                    PointLight() = default;
                    PointLight(Kamanri::Maths::Vector location, double power, RGB color):
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
		}
	}
}