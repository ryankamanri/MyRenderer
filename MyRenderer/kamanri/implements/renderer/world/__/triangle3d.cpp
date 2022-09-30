#include <cfloat>
#include "../../../../renderer/world/__/triangle3d.hpp"
#include "../../../../utils/logs.hpp"
#include "../../../../maths/vector.hpp"
#include "../../../../maths/matrix.hpp"
#include "../../../../utils/string.hpp"

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
                } // namespace Triangle3D$
                
            } // namespace __
            
        } // namespace World
        
    } // namespace Renderer
    
} // namespace Kamanri




Triangle3D::Triangle3D(std::vector<Maths::Vector>& vertices_transform, int offset, int v1, int v2, int v3): _vertices_transform(vertices_transform), _offset(offset), _v1(v1), _v2(v2), _v3(v3)
{

}

void Triangle3D::PrintTriangle(bool is_print) const
{
    if(!is_print) return;
    PrintLn("offset: %d, a: %f, b: %f, c: %f", _offset, _a, _b, _c);
    _po_v1->PrintVector();
    _po_v2->PrintVector();
    _po_v3->PrintVector();
}





void Triangle3D::Build()
{
    _o_i1 = _offset + _v1;
    _o_i2 = _offset + _v2;
    _o_i3 = _offset + _v3;

    _po_v1 = &_vertices_transform[_o_i1];
    _po_v2 = &_vertices_transform[_o_i2];
    _po_v3 = &_vertices_transform[_o_i3];

    _o_v1_x = _po_v1->GetFast(0);
    _o_v1_y = _po_v1->GetFast(1);
    _o_v2_x = _po_v2->GetFast(0);
    _o_v2_y = _po_v2->GetFast(1);
    _o_v3_x = _po_v3->GetFast(0);
    _o_v3_y = _po_v3->GetFast(1);

    SMatrix vertices_matrix = 
    {
        _po_v1->GetFast(0), _po_v1->GetFast(1), _po_v1->GetFast(2),
        _po_v2->GetFast(0), _po_v2->GetFast(1), _po_v2->GetFast(2),
        _po_v3->GetFast(0), _po_v3->GetFast(1), _po_v3->GetFast(2)
    };

    Vector abc_vec = {1, 1, 1};

    (*-vertices_matrix) * abc_vec;

    _a = abc_vec.GetFast(0);
    _b = abc_vec.GetFast(1);
    _c = abc_vec.GetFast(2);
}

#define Determinant(a00, a01, a10, a11) ((a00) * (a11) - (a10) * (a01))

bool Triangle3D::IsIn(double x, double y) const
{

    auto v1_v2_xy_determinant = Determinant
    (
        _o_v2_x - _o_v1_x, x - _o_v1_x,
        _o_v2_y - _o_v1_y, y - _o_v1_y
    );
    auto v2_v3_xy_determinant = Determinant
    (
        _o_v3_x - _o_v2_x, x - _o_v2_x,
        _o_v3_y - _o_v2_y, y - _o_v2_y
    );
    auto v3_v1_xy_determinant = Determinant
    (
        _o_v1_x - _o_v3_x, x - _o_v3_x,
        _o_v1_y - _o_v3_y, y - _o_v3_y
    );

    if (v1_v2_xy_determinant * v2_v3_xy_determinant >= 0 && v2_v3_xy_determinant * v3_v1_xy_determinant >= 0 && v3_v1_xy_determinant * v1_v2_xy_determinant >= 0)
    {
        return true;
    }

    return false;
}

inline double Max(double x1, double x2, double x3)
{
    return x1 > x2 ? (x1 > x3 ? x1 : x3) : (x2 > x3 ? x2 : x3);
}

inline double Min(double x1, double x2, double x3)
{
    return x1 < x2 ? (x1 < x3 ? x1 : x3) : (x2 < x3 ? x2 : x3);
}

bool Triangle3D::GetMinMaxWidthHeight(int &min_width, int &min_height, int &max_width, int &max_height) const
{
    min_width = (int)Min(_o_v1_x, _o_v2_x, _o_v3_x);
    min_height = (int)Min(_o_v1_y, _o_v2_y, _o_v3_y);
    max_width = (int)Max(_o_v1_x, _o_v2_x, _o_v3_x);
    max_height = (int)Max(_o_v1_y, _o_v2_y, _o_v3_y);

    
    return true;
}

Result<Vector> Triangle3D::ArealCoordinates(double x, double y, bool is_print) const
{
    if(!IsIn(x, y))
    {
        Log::Error(__Triangle3D::LOG_NAME, "The target is not in Triangle");
        RESULT_EXCEPTION(Vector, Triangle3D$::CODE_NOT_IN_TRIANGLE, "The target is not in Triangle");
    }
    Vector target = {x, y, Z(x, y), 1};

    Log::Trace("The target is:");
    target.PrintVector(is_print);
    // (v1, v2, v3, 0) (alpha, beta, gamma, 1)^T = target
    SMatrix a = 
    {
        { _o_v1_x, _o_v1_y, _po_v1->GetFast(2), 1},
        { _o_v2_x, _o_v2_y, _po_v2->GetFast(2), 1},
        { _o_v3_x, _o_v3_y, _po_v3->GetFast(2), 1},
        {0, 0, 0, 1}
    };

    Log::Trace("The a is:");
    a.PrintMatrix(is_print);

    TRY_FOR_TYPE(Vector, -a, _a);

    Log::Trace("The -a is:");
    _a.PrintMatrix(is_print);
    ASSERT_FOR_TYPE(Vector, _a * target);
    return Result<Vector>(target);
}