#include "../../renderer/triangle3ds.hpp"
#include "../../utils/logs.hpp"
#include "../../maths/vectors.hpp"
#include "../../maths/matrix.hpp"

using namespace Kamanri::Renderer::Triangle3Ds;
using namespace Kamanri::Utils::Logs;
using namespace Kamanri::Maths::Vectors;
using namespace Kamanri::Maths::Matrix;

constexpr const char* LOG_NAME = "Kamanri::Renderer::Triangle3D";

Triangle3D::Triangle3D(std::vector<Maths::Vectors::Vector>& vertices_transform, int offset, int v1, int v2, int v3): _vertices_transform(vertices_transform), _offset(offset), _v1(v1), _v2(v2), _v3(v3)
{

}

void Triangle3D::PrintTriangle(bool is_print) const
{
    if(!is_print) return;
    PrintLn("offset: %d, a: %f, b: %f, c: %f", _offset, _a, _b, _c);
    _vertices_transform[_offset + _v1].PrintVector();
    _vertices_transform[_offset + _v2].PrintVector();
    _vertices_transform[_offset + _v3].PrintVector();
}

void Triangle3D::Build()
{
    auto v1 = _offset + _v1;
    auto v2 = _offset + _v2;
    auto v3 = _offset + _v3;

    SMatrix vertices_matrix = 
    {
        **_vertices_transform[v1][0], **_vertices_transform[v1][1], **_vertices_transform[v1][2],
        **_vertices_transform[v2][0], **_vertices_transform[v2][1], **_vertices_transform[v2][2],
        **_vertices_transform[v3][0], **_vertices_transform[v3][1], **_vertices_transform[v3][2]
    };

    Vector abc_vec = {1, 1, 1};

    (**-vertices_matrix) * abc_vec;

    _a = **abc_vec[0];
    _b = **abc_vec[1];
    _c = **abc_vec[2];
}

#define Determinant(a00, a01, a10, a11) ((a00) * (a11) - (a10) * (a01))

bool Triangle3D::IsIn(double x, double y)
{
    auto v1 = _offset + _v1;
    auto v2 = _offset + _v2;
    auto v3 = _offset + _v3;

    auto v1_x = _vertices_transform[v1].GetFast(0);
    auto v1_y = _vertices_transform[v1].GetFast(1);
    auto v2_x = _vertices_transform[v2].GetFast(0);
    auto v2_y = _vertices_transform[v2].GetFast(1);
    auto v3_x = _vertices_transform[v3].GetFast(0);
    auto v3_y = _vertices_transform[v3].GetFast(1);



    auto v1_v2_xy_determinant = Determinant
    (
        v2_x - v1_x, x - v1_x,
        v2_y - v1_y, y - v1_y
    );
    auto v2_v3_xy_determinant = Determinant
    (
        v3_x - v2_x, x - v2_x,
        v3_y - v2_y, y - v2_y
    );
    auto v3_v1_xy_determinant = Determinant
    (
        v1_x - v3_x, x - v3_x,
        v1_y - v3_y, y - v3_y
    );

    if (v1_v2_xy_determinant * v2_v3_xy_determinant >= 0 && v2_v3_xy_determinant * v3_v1_xy_determinant >= 0 && v3_v1_xy_determinant * v1_v2_xy_determinant >= 0)
    {
        return true;
    }

    return false;
}

double Triangle3D::Z(double x, double y) const
{
    return (1 - _a * x - _b * y) / _c;
}
