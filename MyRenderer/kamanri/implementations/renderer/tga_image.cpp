#include <iostream>
#include <cstring>
#include "kamanri/renderer/tga_image.hpp"
#include "cuda_dll/exports/memory_operations.hpp"

using namespace Kamanri::Renderer;
using namespace Kamanri::Renderer::TGAImage$;

namespace __TGAImage
{
	constexpr const char* LOG_NAME = STR(TGAImage);

	dll cuda_dll;
	func_type(CUDAMalloc) cuda_malloc;
	func_type(CUDAFree) cuda_free;
	func_type(TransmitToCUDA) transmit_to_cuda;

} // namespace __TGAImage


TGAImage::TGAImage(const int w, const int h, const int bpp) : _width(w), _height(h), _bytes_per_pixel(bpp), _data(w*h*bpp, 0) {}

TGAImage::~TGAImage()
{
	
}

void TGAImage::DeleteCUDA()
{
	__TGAImage::cuda_free(_cuda_data);
}

bool TGAImage::ReadTGAFile(const std::string filename, bool is_use_cuda) {
	std::ifstream in;
	in.open (filename, std::ios::binary);
	if (!in.is_open()) {
		std::cerr << "can't open file " << filename << "\n";
		in.close();
		return false;
	}
	TGAHeader header;
	in.read(reinterpret_cast<char *>(&header), sizeof(header));
	if (!in.good()) {
		in.close();
		std::cerr << "an error occured while reading the header\n";
		return false;
	}
	_width   = header.width;
	_height   = header.height;
	_bytes_per_pixel = header.bits_per_pixel>>3;
	if (_width<=0 || _height<=0 || (_bytes_per_pixel!=GRAYSCALE && _bytes_per_pixel!=RGB && _bytes_per_pixel!=RGBA)) {
		in.close();
		std::cerr << "bad bpp (or width/height) value\n";
		return false;
	}
	size_t nbytes = _bytes_per_pixel*_width*_height;
	_data = std::vector<std::uint8_t>(nbytes, 0);
	if (3==header.data_type_code || 2==header.data_type_code) {
		in.read(reinterpret_cast<char *>(_data.data()), nbytes);
		if (!in.good()) {
			in.close();
			std::cerr << "an error occured while reading the data\n";
			return false;
		}
	} else if (10==header.data_type_code||11==header.data_type_code) {
		if (!LoadRLEData(in)) {
			in.close();
			std::cerr << "an error occured while reading the data\n";
			return false;
		}
	} else {
		in.close();
		std::cerr << "unknown file format " << (int)header.data_type_code << "\n";
		return false;
	}
	if (!(header.image_descriptor & 0x20))
		FlipVertically();
	if (header.image_descriptor & 0x10)
		FlipHorizontally();
	std::cerr << _width << "x" << _height << "/" << _bytes_per_pixel*8 << "\n";
	in.close();

///////////////////////////////////////////////
	if(!is_use_cuda) return true;

	using namespace __TGAImage;
	load_dll(cuda_dll, cuda_dll, LOG_NAME);
	import_func(CUDAMalloc, cuda_dll, cuda_malloc, LOG_NAME);
	import_func(CUDAFree, cuda_dll, cuda_free, LOG_NAME);
	import_func(TransmitToCUDA, cuda_dll, transmit_to_cuda, LOG_NAME);

	auto data_size = _data.size();
	cuda_malloc(&(void*)_cuda_data, data_size * sizeof(std::uint8_t));
	transmit_to_cuda(&_data[0], _cuda_data, data_size * sizeof(std::uint8_t));
	return true;
}

/**
 * @brief Load data in RLE encode
 * 
 * @param in 
 * @return true 
 * @return false 
 */
bool TGAImage::LoadRLEData(std::ifstream &in) {
	size_t pixelcount = _width*_height;
	size_t currentpixel = 0;
	size_t currentbyte  = 0;
	TGAColor colorbuffer;
	do {
		std::uint8_t chunkheader = 0;
		chunkheader = in.get();
		if (!in.good()) {
			std::cerr << "an error occured while reading the data\n";
			return false;
		}
		if (chunkheader<128) {
			chunkheader++;
			for (int i=0; i<chunkheader; i++) {
				in.read(reinterpret_cast<char *>(colorbuffer.bgra), _bytes_per_pixel);
				if (!in.good()) {
					std::cerr << "an error occured while reading the header\n";
					return false;
				}
				for (int t=0; t<_bytes_per_pixel; t++)
					_data[currentbyte++] = colorbuffer.bgra[t];
				currentpixel++;
				if (currentpixel>pixelcount) {
					std::cerr << "Too many pixels read\n";
					return false;
				}
			}
		} else {
			chunkheader -= 127;
			in.read(reinterpret_cast<char *>(colorbuffer.bgra), _bytes_per_pixel);
			if (!in.good()) {
				std::cerr << "an error occured while reading the header\n";
				return false;
			}
			for (int i=0; i<chunkheader; i++) {
				for (int t=0; t<_bytes_per_pixel; t++)
					_data[currentbyte++] = colorbuffer.bgra[t];
				currentpixel++;
				if (currentpixel>pixelcount) {
					std::cerr << "Too many pixels read\n";
					return false;
				}
			}
		}
	} while (currentpixel < pixelcount);
	return true;
}

bool TGAImage::WriteTGAFile(const std::string filename, const bool vflip, const bool rle) const {
	constexpr std::uint8_t developer_area_ref[4] = {0, 0, 0, 0};
	constexpr std::uint8_t extension_area_ref[4] = {0, 0, 0, 0};
	constexpr std::uint8_t footer[18] = {'T','R','U','E','V','I','S','I','O','N','-','X','F','I','L','E','.','\0'};
	std::ofstream out;
	out.open (filename, std::ios::binary);
	if (!out.is_open()) {
		std::cerr << "can't open file " << filename << "\n";
		out.close();
		return false;
	}
	TGAHeader header;
	header.bits_per_pixel = _bytes_per_pixel<<3;
	header.width  = _width;
	header.height = _height;
	header.data_type_code = (_bytes_per_pixel==GRAYSCALE?(rle?11:3):(rle?10:2));
	header.image_descriptor = vflip ? 0x00 : 0x20; // top-left or bottom-left origin
	out.write(reinterpret_cast<const char *>(&header), sizeof(header));
	if (!out.good()) {
		out.close();
		std::cerr << "can't dump the tga file\n";
		return false;
	}
	if (!rle) {
		out.write(reinterpret_cast<const char *>(_data.data()), _width*_height*_bytes_per_pixel);
		if (!out.good()) {
			std::cerr << "can't unload raw data\n";
			out.close();
			return false;
		}
	} else if (!UnloadRLEData(out)) {
			out.close();
			std::cerr << "can't unload rle data\n";
			return false;
		}
	out.write(reinterpret_cast<const char *>(developer_area_ref), sizeof(developer_area_ref));
	if (!out.good()) {
		std::cerr << "can't dump the tga file\n";
		out.close();
		return false;
	}
	out.write(reinterpret_cast<const char *>(extension_area_ref), sizeof(extension_area_ref));
	if (!out.good()) {
		std::cerr << "can't dump the tga file\n";
		out.close();
		return false;
	}
	out.write(reinterpret_cast<const char *>(footer), sizeof(footer));
	if (!out.good()) {
		std::cerr << "can't dump the tga file\n";
		out.close();
		return false;
	}
	out.close();
	return true;
}

// TODO: it is not necessary to break a raw chunk for two equal pixels (for the matter of the resulting size)
bool TGAImage::UnloadRLEData(std::ofstream &out) const {
	const std::uint8_t max_chunk_length = 128;
	size_t npixels = _width*_height;
	size_t curpix = 0;
	while (curpix<npixels) {
		size_t chunkstart = curpix*_bytes_per_pixel;
		size_t curbyte = curpix*_bytes_per_pixel;
		std::uint8_t run_length = 1;
		bool raw = true;
		while (curpix+run_length<npixels && run_length<max_chunk_length) {
			bool succ_eq = true;
			for (int t=0; succ_eq && t<_bytes_per_pixel; t++)
				succ_eq = (_data[curbyte+t]==_data[curbyte+t+_bytes_per_pixel]);
			curbyte += _bytes_per_pixel;
			if (1==run_length)
				raw = !succ_eq;
			if (raw && succ_eq) {
				run_length--;
				break;
			}
			if (!raw && !succ_eq)
				break;
			run_length++;
		}
		curpix += run_length;
		out.put(raw?run_length-1:run_length+127);
		if (!out.good()) {
			std::cerr << "can't dump the tga file\n";
			return false;
		}
		out.write(reinterpret_cast<const char *>(_data.data()+chunkstart), (raw?run_length*_bytes_per_pixel:_bytes_per_pixel));
		if (!out.good()) {
			std::cerr << "can't dump the tga file\n";
			return false;
		}
	}
	return true;
}

TGAColor TGAImage::Get(const int x, const int y) const {
	if (!_data.size() || x<0 || y<0 || x>=_width || y>=_height)
		return {};
	return TGAColor(_data.data()+(x+y*_width)*_bytes_per_pixel, _bytes_per_pixel);
}

TGAColor TGAImage::Get(double u, double v) const
{
	int x = (int)(u * _width);
	int y = (int)(v * _height);
	return Get(x, _height - y); // y axis towards up, so use height - y
}

void TGAImage::Set(int x, int y, const TGAColor &c) {
	if (!_data.size() || x<0 || y<0 || x>=_width || y>=_height) return;
	memcpy(_data.data()+(x+y*_width)*_bytes_per_pixel, c.bgra, _bytes_per_pixel);
}



void TGAImage::FlipHorizontally() {
	int half = _width>>1;
	for (int i=0; i<half; i++)
		for (int j=0; j<_height; j++)
			for (int b=0; b<_bytes_per_pixel; b++)
				std::swap(_data[(i+j*_width)*_bytes_per_pixel+b], _data[(_width-1-i+j*_width)*_bytes_per_pixel+b]);
}

void TGAImage::FlipVertically() {
	int half = _height>>1;
	for (int i=0; i<_width; i++)
		for (int j=0; j<half; j++)
			for (int b=0; b<_bytes_per_pixel; b++)
				std::swap(_data[(i+j*_width)*_bytes_per_pixel+b], _data[(i+(_height-1-j)*_width)*_bytes_per_pixel+b]);
}

int TGAImage::Width() const {
	return _width;
}

int TGAImage::Height() const {
	return _height;
}


