#pragma once
#include "kamanri/renderer/world/__/buffers.hpp"

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
					__device__ inline size_t Scan_R270(size_t height, size_t x, size_t y)
					{
						return ((height - (y + 1)) * height + x);
					}
				}
			}
		}
	}
}


__device__ Kamanri::Renderer::World::FrameBuffer& Kamanri::Renderer::World::__::Buffers::GetFrame(size_t x, size_t y)
{
	using namespace __Buffers;
	if (x >= _width || y >= _height)
	{
		Kamanri::Utils::PrintLn("Invalid Index (%llu, %llu), (width, height) = (%llu, %llu), return the 0 index content\n", x, y, _width, _height);
		return _cuda_buffers[0];
	}
	return _cuda_buffers[Scan_R270(_height, x, y)];

}

__device__ void Kamanri::Renderer::World::__::Buffers::InitPixel(size_t x, size_t y)
{
	auto& frame = GetFrame(x, y);
	frame.location = Kamanri::Maths::Vector(4);
	frame.location.Set(2, -DBL_MAX);
	frame.vertex_normal = Kamanri::Maths::Vector(4);
	
	auto& bitmap = GetBitmapBuffer(x, y);
	bitmap = 0x0;
}

__device__ DWORD& Kamanri::Renderer::World::__::Buffers::GetBitmapBuffer(size_t x, size_t y)
{
	using namespace __Buffers;
	if (x >= _width || y >= _height)
	{
		Kamanri::Utils::PrintLn("Invalid Index (%llu, %llu), (width, height) = (%llu, %llu), return the 0 index content\n", x, y, _width, _height);
		return _cuda_bitmap_buffer[0];
	}
	return _cuda_bitmap_buffer[Scan_R270(_height, x, y)]; // (x, y) -> (x, _height - y)
}