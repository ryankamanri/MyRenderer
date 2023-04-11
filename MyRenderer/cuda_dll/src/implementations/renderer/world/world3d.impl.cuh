#pragma once
#include "kamanri/renderer/world/world3d.hpp"

__device__ void Kamanri::Renderer::World::World3D::__BuildForPixel(size_t x, size_t y)
{
	// set z = infinity
	_buffers.InitPixel(x, y);

	auto& buffer = _buffers.GetFrame(x, y);
	auto& bitmap_pixel = _buffers.GetBitmapBuffer(x, y);

	for (size_t i = 0; i < _environment.cuda_triangles.size; i++)
	{
		_environment.cuda_triangles.data[i].WriteToPixel(x, y, buffer, _camera.NearestDist(), _environment.cuda_objects.data);
	}

	if (_buffers.GetFrame(x, y).location[2] == -DBL_MAX) return;

	// set distance = infinity, is exposed.
	_environment.bpr_model.InitLightBufferPixel(x, y, buffer);

	// while(1);

	// for (size_t i = 0; i < _environment.cuda_triangles.size; i++)
	// {
	// 	_environment.bpr_model.__BuildPerTrianglePixel(x, y, _environment.cuda_triangles.data[i], buffer);
	// }


	_environment.bpr_model.__BuildPixel(x, y, _environment.cuda_triangles, _environment.cuda_boxes.data, buffer);

	_environment.bpr_model.WriteToPixel(x, y, buffer, bitmap_pixel);


}