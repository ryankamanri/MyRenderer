#pragma once
#include "kamanri/renderer/world/world3d.hpp"

__device__ void Kamanri::Renderer::World::World3D::__BuildForPixel(size_t x, size_t y)
{
	// set z = infinity
	_buffers.InitPixel(x, y);

	auto& buffer = _buffers.GetFrame(x, y);
	auto& bitmap_pixel = _buffers.GetBitmapBuffer(x, y);

	__::BoundingBox$::MayScreenCover(
		_environment.cuda_boxes.data,
		0, _environment.cuda_triangles,
		x, y,
		[](
			__::Triangle3D& triangle,
			size_t x,
			size_t y,
			FrameBuffer& buffer,
			double nearest_dist,
			Object* cuda_objects)
	{
		triangle.WriteToPixel(x, y, buffer, nearest_dist, cuda_objects);
	}, buffer, _camera.NearestDist(), _environment.cuda_objects.data);

	if (_buffers.GetFrame(x, y).location[2] == -DBL_MAX) return;

	// set distance = infinity, is exposed.
	_environment.bpr_model.InitLightBufferPixel(x, y, buffer);


	_environment.bpr_model.__BuildPixel(x, y, _environment.cuda_triangles, _environment.cuda_boxes.data, buffer);

	_environment.bpr_model.WriteToPixel(x, y, buffer, bitmap_pixel);


}