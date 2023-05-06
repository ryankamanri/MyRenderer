#pragma once

#ifndef SWIG
#include <cstdint>
#include <fstream>
#include <vector>
#endif

namespace Kamanri
{
	namespace Renderer
	{
		namespace TGAImage$
		{
			// This pragma preprocessive instruct is used to align memory
#pragma pack(push, 1)
			struct TGAHeader
			{
				// 图像信息长度、颜色表类型、图像类型码
				std::uint8_t id_length{};      // 指出图像信息字段长度，取值范围0~255
				std::uint8_t color_map_type{}; // 0：不使用颜色表 1：使用颜色表
				std::uint8_t data_type_code{}; // 0：没有图像数据 1：未压缩的颜色表图像 2：未压缩的真彩图像 3：未压缩的黑白图像 9：RLE压缩的颜色表图像 10：RLE压缩的真彩图像 11：RLE压缩的黑白图像
				// 颜色表规格字段
				std::uint16_t color_map_origin{}; // 颜色表首址	2	颜色表首的入口索引
				std::uint16_t color_map_length{}; // 颜色表长度	2	颜色表表项总数
				std::uint8_t color_map_depth{};   // 颜色表项位数	1	位数，16代表16位TGA，24代表24位TGA，32代表32位TGA
				//图像规格字段
				std::uint16_t x_origin{};      // 图像X坐标起始位置	2	图像左下角X坐标
				std::uint16_t y_origin{};      // 图像Y坐标起始位置	2	图像左下角Y坐标
				std::uint16_t width{};         // 图像宽度	2	以像素为单位
				std::uint16_t height{};        // 图像高度	2	以像素为单位
				std::uint8_t bits_per_pixel{}; // 图像每像素存储占用位数	2	值为8、16、24、32等
				std::uint8_t image_descriptor{};
			};
#pragma pack(pop)

			struct TGAColor
			{
				std::uint8_t bgra[4] = { 0, 0, 0, 0 };
				std::uint8_t bytespp = { 0 };
				unsigned int bgr = 0;
				unsigned int rgb = 0;

				TGAColor() = default;
				TGAColor(const std::uint8_t R, const std::uint8_t G, const std::uint8_t B, const std::uint8_t A = 255) : bgra{ B, G, R, A }, bytespp(4) {}
#ifdef __CUDA_RUNTIME_H__  
				__device__
#endif
					TGAColor(const std::uint8_t* p, const std::uint8_t bpp) : bytespp(bpp)
				{
					for (int i = bpp; i--; bgra[i] = p[i]);
					bgr |= bgra[0] << 16;
					bgr |= bgra[1] << 8;
					bgr |= bgra[2];

					rgb |= bgra[2] << 16;
					rgb |= bgra[1] << 8;
					rgb |= bgra[0];
				}
				std::uint8_t& operator[](const int i) { return bgra[i]; }
			};
		} // namespace TGAImage$
	}
}