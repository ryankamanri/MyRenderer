#pragma once
#include "kamanri/renderer/tga_image.hpp"

__device__ Kamanri::Renderer::TGAImage$::TGAColor Kamanri::Renderer::TGAImage::Get(const int x, const int y) const
{
	if (x < 0 || y < 0 || x >= _width || y >= _height)
		return {};
	return TGAImage$::TGAColor(_cuda_data + (x + y * _width) * _bytes_per_pixel, _bytes_per_pixel);
}
__device__ Kamanri::Renderer::TGAImage$::TGAColor Kamanri::Renderer::TGAImage::Get(double u, double v) const
{
	int x = u * _width;
	int y = v * _height;
	return Get(x, _height - y); // y axis towards up, so use height - y
}
