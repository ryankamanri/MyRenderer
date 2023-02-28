#pragma once
#include "kamanri/utils/memory.hpp"
#include "kamanri/renderer/world/frame_buffer.hpp"
// #include "triangle3d.hpp"


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
					size_t _width;
					size_t _height;
					Utils::P<FrameBuffer[]> _buffers;
					FrameBuffer* _cuda_buffers;
					Utils::P<DWORD[]> _bitmap_buffer;
					DWORD* _cuda_bitmap_buffer;

				public:
					Buffers(size_t width, size_t height);
					~Buffers();
					Buffers& operator=(Buffers&& other);
					void InitPixel(size_t x, size_t y);
					void CleanBitmap() const;
					// void WriteFrom(Triangle3D const &t, double nearest_dist);
					inline size_t Width() const { return _width; }
					inline size_t Height() const { return _height; }
					FrameBuffer& GetFrame(size_t x, size_t y);
					inline DWORD* GetBitmapBufferPtr() { return _bitmap_buffer.get(); }
					DWORD& GetBitmapBuffer(size_t x, size_t y);
					inline FrameBuffer* CUDAGetBuffersPtr() { return _cuda_buffers; }
					inline DWORD* CUDAGetBitmapBufferPtr() { return _cuda_bitmap_buffer; }
				};

			} // namespace __

		} // namespace World

	} // namespace Renderer

} // namespace Kamanri

