#include <cmath>
#include "../../../maths/math.hpp"
#include "../../../utils/logs.hpp"
#include "../../../renderer/world/camera.hpp"
#include "../../../maths/matrix.hpp"
#include "../../../utils/string.hpp"

using namespace Kamanri::Utils;
using namespace Kamanri::Renderer::World;
using namespace Kamanri::Renderer::World::Camera$;
using namespace Kamanri::Maths;


namespace Kamanri
{
    namespace Renderer
    {
        namespace World
        {
            namespace __Camera
            {
                constexpr const char *LOG_NAME = STR(Kamanri::Renderer::World::Camera);

                /**
                 * @brief FIxed asin, filtered the situation of x > 1 and x < -1
                 *
                 * @param x
                 * @return double
                 */
                inline double Arcsin(double x)
                {
                    return x > 1 ? asin(1) : (x < -1 ? asin(-1) : asin(x));
                }

                // Camera::Transform
                SMatrix a21(4);
                SMatrix a543(4);

                // Camera::InverseUpperWithDirection
                Vector upward_before = {0, 1, 0, 0};
                Vector upward_after = {0, 1, 0, 0};
            } // namespace __Camera
            
        } // namespace World
        
    } // namespace Renderer
    
} // namespace Kamanri

using namespace Kamanri::Renderer::World::__Camera;


Camera::Camera(Vector location, Vector direction, Vector upper, double nearer_dest, double further_dest, unsigned int screen_width, unsigned int screen_height) : _nearer_dest(nearer_dest), _further_dest(further_dest), _screen_width(screen_width), _screen_height(screen_height)
{
    if (*location.N() != 4 || *direction.N() != 4 || *upper.N() != 4)
    {
        Log::Error(__Camera::LOG_NAME, "Invalid vector length. location length: %d, direction length: %d, upper length: %d",
                   *location.N(), *direction.N(), *upper.N());
        return;
    }

    if (*location[3] != 1 || *direction[3] != 0 || *upper[3] != 0)
    {
        Log::Error(__Camera::LOG_NAME, "Invalid vector type. valid location/direction/upper type: 1/0/0 but given %.0f/%.0f/%.0f", *location[3], *direction[3], *upper[3]);
        return;
    }

    _location = location;
    auto dir_unit_res = direction.Unitization();
    if (dir_unit_res.IsException())
    {
        Log::Error(__Camera::LOG_NAME, "Failed to unitization direction caused by:");
        dir_unit_res.Print();
    }
    _direction = direction;

    auto upp_set_res = upper.Set(2, 0);
    auto upp_unit_res = upper.Unitization();
    if (upp_set_res.IsException() || upp_unit_res.IsException())
    {
        Log::Error(__Camera::LOG_NAME, "Failed to set or unitization upper caused by:");
        upp_set_res.Print();
        upp_unit_res.Print();
    }
    _upward = upper;

    SetAngles();
}



void Camera::SetAngles(bool is_print)
{
    _beta = Arcsin(_direction.GetFast(1)); // beta (-PI/2, PI/2)

    _alpha = Arcsin((_direction.GetFast(0)) / cos(_beta)); // alpha (-PI, PI)

    if(_direction.GetFast(2) > 0)
    {
        if(_alpha > 0) 
            _alpha = PI - _alpha;
        else
            _alpha = -PI - _alpha;
    }

    _gamma = Arcsin(_upward.GetFast(0)); // gamma (-PI, PI)

    if(_upward.GetFast(1) < 0)
    {
        if(_gamma > 0)
            _gamma = PI - _gamma;
        else
            _gamma = -PI - _gamma;
    }

    Log::Trace(__Camera::LOG_NAME, "The direction vector:");

    _direction.PrintVector(is_print);

    Log::Trace(__Camera::LOG_NAME, "alpha = %.2f, beta = %.2f, gamma = %.2f",
        _alpha, _beta, _gamma);
}

Camera::Camera(Camera const &camera) : _pvertices(camera._pvertices), _alpha(camera._alpha), _beta(camera._beta), _gamma(camera._gamma), _nearer_dest(camera._nearer_dest), _further_dest(camera._further_dest), _screen_width(camera._screen_width), _screen_height(camera._screen_height)
{
    _location = *camera._location.Copy();
    _direction = *camera._direction.Copy();
}

void Camera::SetVertices(std::vector<Maths::Vector> &vertices, std::vector<Maths::Vector> &vertices_transform)
{
    _pvertices = &vertices;
    _pvertices_transform = &vertices_transform;
}

#define min(a,b) (((a) < (b)) ? (a) : (b))

DefaultResult Camera::Transform(bool is_print)
{
    CHECK_MEMORY_FOR_DEFAULT_RESULT(_pvertices, __Camera::LOG_NAME, Camera$::CODE_NULL_POINTER_PVERTICES)

    SetAngles(is_print);

    Log::Trace(__Camera::LOG_NAME, "vertices count: %d", _pvertices->size());
    //
    auto sin_a = sin(_alpha);
    auto cos_a = cos(_alpha);
    auto sin_b = sin(_beta);
    auto cos_b = cos(_beta);
    auto sin_a_sin_b = sin_a * sin_b;
    auto sin_a_cos_b = sin_a * cos_b;
    auto cos_a_sin_b = cos_a * sin_b;
    auto cos_a_cos_b = cos_a * cos_b;
    auto sin_g = sin(_gamma);
    auto cos_g = cos(_gamma);
    auto lx = _location.GetFast(0);
    auto ly = _location.GetFast(1);
    auto lz = _location.GetFast(2);


    Log::Trace(__Camera::LOG_NAME, "a5*4*3 * a2*1 * v:");

    // SMatrix a1 = 
    // {
    //     1, 0, 0, -lx,
    //     0, 1, 0, -ly,
    //     0, 0, 1, -lz,
    //     0, 0, 0, 1  
    // };

    // SMatrix a2 = 
    // {
    //     cos_a, 0, sin_a, 0,
    //     -sin_a_sin_b, cos_b, cos_a_sin_b, 0,
    //     -sin_a_cos_b, -sin_b, cos_a_cos_b, 0,
    //     0, 0, 0, 1
    // };

    a21.Reset
    ({
        cos_a, 0, sin_a, -lx * cos_a - lz * sin_a,
        -sin_a_sin_b, cos_b, cos_a_sin_b, lx * sin_a_sin_b - ly * cos_b - lz * cos_a_sin_b,
        -sin_a_cos_b, -sin_b, cos_a_cos_b, lx * sin_a_cos_b + ly * sin_b - lz * cos_a_cos_b,
        0, 0, 0, 1
    });

    // SMatrix a3 = 
    // {
    //     cos_g, -sin_g, 0, 0,
    //     sin_g, cos_g, 0, 0,
    //     0, 0, 1, 0,
    //     0, 0, 0, 1
    // };

    // SMatrix a4 = 
    // {
    //     _nearer_dest, 0, 0, 0,
    //     0, _nearer_dest, 0, 0,
    //     0, 0, _nearer_dest + _further_dest, -_nearer_dest * _further_dest,
    //     0, 0, 1, 0 
    // };

    // SMatrix a5 = 
    // {
    //     _screen_width / 2, 0, 0, _screen_width / 2,
    //     0, -_screen_height / 2, 0, _screen_height / 2,
    //     0, 0, 1, 0,
    //     0, 0, 0, 1
    // };

    // SMatrix a5 = 
    // {
    //     _screen_width / 2, 0, 0, _screen_width / 2,
    //     0, -_screen_height / 2, 0, _screen_height / 2,
    //     0, 0, min(_screen_width, _screen_height) / 2, 0,
    //     0, 0, 0, 1
    // };

    a543.Reset
    ({
        (double)_screen_width * _nearer_dest * cos_g / 2, -(double)_screen_width * _nearer_dest * sin_g / 2, (double)_screen_width / 2, 0,
        -(double)_screen_height * _nearer_dest * sin_g / 2, -(double)_screen_height * _nearer_dest * cos_g / 2, (double)_screen_height / 2, 0,
        0, 0, (_nearer_dest + _further_dest) * min(_screen_width, _screen_height) / 2, (-_nearer_dest * _further_dest) * min(_screen_width, _screen_height) / 2,
        0, 0, 1, 0
    });

    // copy vertices_transform from vertices(origin) and transform it.

    for(auto i = 0; i != _pvertices_transform->size(); i++)
    {
        _pvertices_transform->at(i).CopyFrom(_pvertices->at(i));

        Log::Trace(__Camera::LOG_NAME, "Start a vertex transform...");
        
        _pvertices_transform->at(i).PrintVector(is_print);
        a21 * _pvertices_transform->at(i);
        _pvertices_transform->at(i).PrintVector(is_print);
        a543 * _pvertices_transform->at(i);
        _pvertices_transform->at(i).PrintVector(is_print);
        _pvertices_transform->at(i) *= (1 / (_pvertices_transform->at(i).GetFast(3)));
        _pvertices_transform->at(i).PrintVector(is_print);
    }
    //
    return DEFAULT_RESULT;
}

DefaultResult Camera::InverseUpperWithDirection(Maths::Vector const &last_direction)
{
    auto last_direction_n = *last_direction.N();
    if (last_direction_n != 4)
    {
        Log::Error(__Camera::LOG_NAME, "Invalid last_direction length %d", last_direction_n);
        return DEFAULT_RESULT_EXCEPTION(Camera$::CODE_INVALID_VECTOR_LENGTH, "Invalid last_direction length");
    }

    upward_before.Reset({0, 1, 0, 0});
    upward_after.Reset({0, 1, 0, 0});

    upward_before *= last_direction;
    upward_after *= _direction;

    if (*(upward_before * upward_after) < 0)
    {
        _upward *= -1;
    }

    return DEFAULT_RESULT;
}