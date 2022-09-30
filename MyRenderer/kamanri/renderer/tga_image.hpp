#pragma once
#include <cstdint>
#include <fstream>
#include <vector>

namespace Kamanri
{
    namespace Renderer
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
            std::uint8_t bgra[4] = {0, 0, 0, 0};
            std::uint8_t bytespp = {0};
            unsigned int bgr = 0;

            TGAColor() = default;
            TGAColor(const std::uint8_t R, const std::uint8_t G, const std::uint8_t B, const std::uint8_t A = 255) : bgra{B, G, R, A}, bytespp(4) {}
            TGAColor(const std::uint8_t *p, const std::uint8_t bpp) : bytespp(bpp)
            {
                // for (int i = bpp; i--; bgra[i] = p[i])
                //     ;
                for (int i = bpp; i--; bgra[i] = p[i], bgr |= (p[i] << 8 * (3 - i)))
                    ;
                bgr = bgr >> 8;
            }
            std::uint8_t &operator[](const int i) { return bgra[i]; }
        };

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
            bool ReadTGAFile(const std::string filename);
            bool WriteTGAFile(const std::string filename, const bool vflip = true, const bool rle = true) const;
            void FlipHorizontally();
            void FlipVertically();
            TGAColor Get(const int x, const int y) const;
            void Set(const int x, const int y, const TGAColor &c);
            int Width() const;
            int Height() const;

        private:
            bool LoadRLEData(std::ifstream &in);
            bool UnloadRLEData(std::ofstream &out) const;

            int _width = 0;
            int _height = 0;
            int _bytes_per_pixel = 0;
            std::vector<std::uint8_t> _data = {};
        };
    }
}
