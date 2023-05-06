#pragma once
#ifndef SWIG
#include "kamanri/renderer/tga_image$.hpp"
#endif

namespace Kamanri
{
	namespace Renderer
	{

		struct TGAImage
		{
			enum Format
			{
				GRAYSCALE = 1,
				RGB = 3,
				RGBA = 4
			};

			TGAImage() = default;
			TGAImage(const int w, const int h, const int bpp);
			~TGAImage();
			void DeleteCUDA();
			bool ReadTGAFile(const std::string filename, bool is_use_cuda = false);
			bool WriteTGAFile(const std::string filename, const bool vflip = true, const bool rle = true) const;
			void FlipHorizontally();
			void FlipVertically();
#ifdef __CUDA_RUNTIME_H__  
			__device__
#endif
			Kamanri::Renderer::TGAImage$::TGAColor Get(const int x, const int y) const;
			/// @brief Get with UV coordinates.
#ifdef __CUDA_RUNTIME_H__  
			__device__
#endif
			Kamanri::Renderer::TGAImage$::TGAColor Get(double u, double v) const;
			void Set(const int x, const int y, const Kamanri::Renderer::TGAImage$::TGAColor& c);

			int Width() const;
			int Height() const;

			private:
			bool LoadRLEData(std::ifstream& in);
			bool UnloadRLEData(std::ofstream& out) const;

			int _width = 0;
			int _height = 0;
			int _bytes_per_pixel = 0;
			std::vector<std::uint8_t> _data = {};
			unsigned char* _cuda_data;
		};
	}
}
