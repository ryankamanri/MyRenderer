#include <cfloat>
#include "kamanri/renderer/world/__/triangle3d.hpp"
#include "kamanri/utils/logs.hpp"
#include "kamanri/maths/vector.hpp"
#include "kamanri/maths/matrix.hpp"
#include "kamanri/utils/string.hpp"

using namespace Kamanri::Utils;
using namespace Kamanri::Renderer::World::__;
using namespace Kamanri::Maths;
using namespace Kamanri::Maths;

namespace Kamanri
{
    namespace Renderer
    {
        namespace World
        {
            namespace __
            {
                namespace __Triangle3D
                {
                    constexpr const char* LOG_NAME = STR(Kamanri::Renderer::World::__::Triangle3D);

                    namespace IsCover
                    {
                        double v1_v2_xy_determinant;
                        double v2_v3_xy_determinant;
                        double v3_v1_xy_determinant;
                    } // namespace IsCover
                    
                    namespace Build
                    {
                        SMatrix vertices_matrix(3);
                        Vector abc_vec(3);
                    } // namespace Build
                    
                    namespace WritePixelTo
                    {
                        Result<Maths::Vector> res;
                        double areal_coordinates_1;
                        double areal_coordinates_2;
                        double areal_coordinates_3;

                        double world_z;

                        double img_u;
                        double img_v;
                    } // namespace WritePixelTo

                    namespace WriteTo
                    {
                        double nearest_dist;
                    } // namespace WriteTo
                    
                    

                    namespace ArealCoordinates
                    {
                        Vector target(4);
                        SMatrix a(4);
                        SMatrix n_a(4); //negative a.
                    } // namespace ArealCoordinates
                    
                } // namespace Triangle3D$
                
            } // namespace __
            
        } // namespace World
        
    } // namespace Renderer
    
} // namespace Kamanri




Triangle3D::Triangle3D(Object& object, int v1, int v2, int v3, int vt1, int vt2, int vt3, int vn1, int vn2, int vn3)
{
    _p_object = &object;
    _v1 = v1;
    _v2 = v2;
    _v3 = v3;
    _vt1 = vt1;
    _vt2 = vt2;
    _vt3 = vt3;
    _vn1 = vn1;
    _vn2 = vn2;
    _vn3 = vn3;

}

void Triangle3D::PrintTriangle(LogLevel level) const
{
    if(Log::Level() > level) return;
    PrintLn("v1 | v2 | v3 : %d | %d | %d", _v1, _v2, _v3);
}


void Triangle3D::Build(Resources const& res)
{
    using namespace __Triangle3D::Build;

    // 1. Build the location of triangle
    _v1_x = res.vertices_transformed[_v1].GetFast(0);
    _v1_y = res.vertices_transformed[_v1].GetFast(1);
    _v1_z = res.vertices_transformed[_v1].GetFast(2);
    _v2_x = res.vertices_transformed[_v2].GetFast(0);
    _v2_y = res.vertices_transformed[_v2].GetFast(1);
    _v2_z = res.vertices_transformed[_v2].GetFast(2);
    _v3_x = res.vertices_transformed[_v3].GetFast(0);
    _v3_y = res.vertices_transformed[_v3].GetFast(1);
    _v3_z = res.vertices_transformed[_v3].GetFast(2);

    // add world z
    _w_v1_z = res.vertices_model_view_transformed[_v1].GetFast(2);
    _w_v2_z = res.vertices_model_view_transformed[_v2].GetFast(2);
    _w_v3_z = res.vertices_model_view_transformed[_v3].GetFast(2);


    vertices_matrix = 
    {
        _v1_x, _v1_y, _v1_z,
        _v2_x, _v2_y, _v2_z,
        _v3_x, _v3_y, _v3_z
    };

    abc_vec = {1, 1, 1};

    (*-vertices_matrix) * abc_vec;

    Log::Trace(__Triangle3D::LOG_NAME, "The abc_vec is:");
    abc_vec.PrintVector(Log$::TRACE_LEVEL);

    _a = abc_vec.GetFast(0);
    _b = abc_vec.GetFast(1);
    _c = abc_vec.GetFast(2);

    // 2. Build the color of every pixel in triangle
    _vt1_x = res.vertex_textures[_vt1].GetFast(0);
    _vt1_y = res.vertex_textures[_vt1].GetFast(1);
    _vt2_x = res.vertex_textures[_vt2].GetFast(0);
    _vt2_y = res.vertex_textures[_vt2].GetFast(1);
    _vt3_x = res.vertex_textures[_vt3].GetFast(0);
    _vt3_y = res.vertex_textures[_vt3].GetFast(1);

}

#define Determinant(a00, a01, a10, a11) ((a00) * (a11) - (a10) * (a01))

bool Triangle3D::IsCover(double x, double y) const
{
    using namespace __Triangle3D::IsCover;
    v1_v2_xy_determinant = Determinant
    (
        _v2_x - _v1_x, x - _v1_x,
        _v2_y - _v1_y, y - _v1_y
    );
    v2_v3_xy_determinant = Determinant
    (
        _v3_x - _v2_x, x - _v2_x,
        _v3_y - _v2_y, y - _v2_y
    );
    v3_v1_xy_determinant = Determinant
    (
        _v1_x - _v3_x, x - _v3_x,
        _v1_y - _v3_y, y - _v3_y
    );

    if (v1_v2_xy_determinant * v2_v3_xy_determinant >= 0 && v2_v3_xy_determinant * v3_v1_xy_determinant >= 0 && v3_v1_xy_determinant * v1_v2_xy_determinant >= 0)
    {
        return true;
    }

    return false;
}

void Triangle3D::WritePixelTo(double x, double y, FrameBuffer& frame_buffer) const
{
    using namespace __Triangle3D::WritePixelTo;
    res = std::move(ArealCoordinates(x, y));
    if(res.IsException())
    {
        Log::Error(__Triangle3D::LOG_NAME, "Cannot get the ArealCoordinates.");
        res.Print();
    }

    areal_coordinates_1 = res.Data().GetFast(0);
    areal_coordinates_2 = res.Data().GetFast(1);
    areal_coordinates_3 = res.Data().GetFast(2);

    // *****************************************
    // 透视矫正
    // *****************************************
    // img_u = areal_coordinates_1 * _vt1_x + areal_coordinates_2 * _vt2_x + areal_coordinates_3 * _vt3_x;
    // img_v = areal_coordinates_1 * _vt1_y + areal_coordinates_2 * _vt2_y + areal_coordinates_3 * _vt3_y;

    world_z = 1.0 / (areal_coordinates_1 / _w_v1_z + areal_coordinates_2 / _w_v2_z + areal_coordinates_3 / _w_v3_z);

    if(world_z < frame_buffer.z || world_z > __Triangle3D::WriteTo::nearest_dist) return;

    img_u = (areal_coordinates_1 * _vt1_x / _w_v1_z + areal_coordinates_2 * _vt2_x / _w_v2_z + areal_coordinates_3 * _vt3_x / _w_v3_z) * world_z; 
    img_v = (areal_coordinates_1 * _vt1_y / _w_v1_z + areal_coordinates_2 * _vt2_y / _w_v2_z + areal_coordinates_3 * _vt3_y / _w_v3_z) * world_z;
    
    frame_buffer.z = world_z;
    frame_buffer.color = _p_object->GetImage().Get(img_u, img_v).bgr;
}

inline double Max(double x1, double x2, double x3)
{
    return x1 > x2 ? (x1 > x3 ? x1 : x3) : (x2 > x3 ? x2 : x3);
}

inline double Min(double x1, double x2, double x3)
{
    return x1 < x2 ? (x1 < x3 ? x1 : x3) : (x2 < x3 ? x2 : x3);
}

void Triangle3D::WriteTo(Buffers& buffers, double nearest_dist)
{
    __Triangle3D::WriteTo::nearest_dist = nearest_dist;

    int min_width = (int)Min(_v1_x, _v2_x, _v3_x);
    int min_height = (int)Min(_v1_y, _v2_y, _v3_y);
    int max_width = (int)Max(_v1_x, _v2_x, _v3_x);
    int max_height = (int)Max(_v1_y, _v2_y, _v3_y);

    for (int i = min_width; i <= max_width; i++)
    {

        if (i >= buffers.Width() || i < 0)
            continue;
        for (int j = min_height; j <= max_height; j++)
        {
            if (j >= buffers.Height() || j < 0)
                continue;

            if(!IsCover(i, j)) 
                continue;
            
            // Now the point is on the triangle
            // start to compare the depth
            WritePixelTo(i, j, buffers.Get(i, j));   
                
        }
    }
}


Result<Vector> Triangle3D::ArealCoordinates(double x, double y) const
{
    if(!IsCover(x, y))
    {
        Log::Error(__Triangle3D::LOG_NAME, "The target is not in Triangle");
        RESULT_EXCEPTION(Vector, Triangle3D$::CODE_NOT_IN_TRIANGLE, "The target is not in Triangle");
    }
    using namespace __Triangle3D::ArealCoordinates;
    target = {x, y, ScreenZ(x, y), 1};

    Log::Trace(__Triangle3D::LOG_NAME,"The target is:");
    target.PrintVector(Log$::TRACE_LEVEL);
    // (v1, v2, v3, 0) (alpha, beta, gamma, 1)^T = target
    a = 
    {
        { _v1_x, _v1_y, _v1_z, 1},
        { _v2_x, _v2_y, _v2_z, 1},
        { _v3_x, _v3_y, _v3_z, 1},
        {0, 0, 0, 1}
    };

    Log::Trace(__Triangle3D::LOG_NAME, "The a is:");
    a.PrintMatrix(Log$::TRACE_LEVEL);

    n_a = TRY_FOR_TYPE(Vector, -a);

    Log::Trace(__Triangle3D::LOG_NAME, "The -a is:");
    n_a.PrintMatrix(Log$::TRACE_LEVEL);
    ASSERT_FOR_TYPE(Vector, n_a * target);
    return Result<Vector>(target);
}