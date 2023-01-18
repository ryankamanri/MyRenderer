#pragma once
#include <vector>
#include "kamanri/maths/vector.hpp"
#include "kamanri/renderer/tga_image.hpp"
#include "kamanri/renderer/world/object.hpp"
#include "resources.hpp"
namespace Kamanri
{
    namespace Renderer
    {
        namespace World
        {
            namespace __
            {

                namespace Triangle3D$
                {
                    constexpr int CODE_NOT_IN_TRIANGLE = 100;
                    constexpr int INEXIST_INDEX = -1;
                } // namespace Triangle3D$
                
                /**
                 * @brief The Triangle3D class is designed to represent a triangle consist of 3 vertices,
                 * it will be the object of projection transformation and be rendered to screen.
                 *
                 */
                class Triangle3D
                {
                private:
                    /// @brief The object triangle belongs to.
                    Object& _object;

                    // offset + index
                    int _v1, _v2, _v3;
                    int _vt1, _vt2, _vt3;
                    int _vn1, _vn2, _vn3;
                    
                    /// values
                    // on screen coordinates
                    double _v1_x, _v1_y, _v1_z, _v2_x, _v2_y, _v2_z, _v3_x, _v3_y, _v3_z;
                    // on world coordinates
                    double _w_v1_z, _w_v2_z, _w_v3_z;
                    double _vt1_x, _vt1_y, _vt2_x, _vt2_y, _vt3_x, _vt3_y;
                    double _vn1_x, _vn1_y, _vn1_z, _vn2_x, _vn2_y, _vn2_z, _vn3_x, _vn3_y, _vn3_z;

                    // factors of square, ax + by + cz - 1 = 0
                    // DEPRECATED
                    double _a, _b, _c;

                public:
                    Triangle3D(Object& object, int v1, int v2, int v3, int vt1, int vt2, int vt3, int vn1, int vn2, int vn3);
                    void Build(Resources const& res, bool is_print = false);
                    bool Cover(double x, double y) const;
                    inline double Z(double x, double y) const { return (1 - _a * x - _b * y) / _c; }
                    unsigned int Color(double x, double y, bool is_print = false) const;
                    void PrintTriangle(bool is_print = true) const;
                    bool GetMinMaxWidthHeight(int &min_width, int &min_height, int &max_width, int &max_height) const;
                    Utils::Result<Maths::Vector> ArealCoordinates(double x, double y, bool is_print = false) const;
                };
            } // namespace __

        } // namespace Triangle3Ds

    } // namespace Renderer

} // namespace Kamanri