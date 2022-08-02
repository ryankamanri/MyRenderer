#include <math.h>
#include "../../utils/logs.hpp"
#include "../../renderer/cameras.hpp"
#include "../../maths/matrix.hpp"

using namespace Kamanri::Renderer::Cameras;
using namespace Kamanri::Utils::Logs;
using namespace Kamanri::Maths::Vectors;
using namespace Kamanri::Maths::Matrix;
using namespace Kamanri::Utils::Memory;
using namespace Kamanri::Utils::Result;

constexpr const char *LOG_NAME = "Kamanri::Renderer::Cameras";

Camera::Camera(Vector location, Vector direction, Vector upper, double nearer_dest, double further_dest, unsigned int screen_width, unsigned int screen_height) : _nearer_dest(nearer_dest), _further_dest(further_dest), _screen_width(screen_width), _screen_height(screen_height)
{
    if (**location.N() != 4 || **direction.N() != 4 || **upper.N() != 4)
    {
        Log::Error(LOG_NAME, "Invalid vector length. location length: %d, direction length: %d, upper length: %d",
                   **location.N(), **direction.N(), **upper.N());
        return;
    }

    if (**location[3] != 1 || **direction[3] != 0 || **upper[3] != 0)
    {
        Log::Error(LOG_NAME, "Invalid vector type. valid location/direction/upper type: 1/0/0 but given %.0f/%.0f/%.0f", **location[3], **direction[3], **upper[3]);
        return;
    }

    _location = location;
    auto dir_unit_res = direction.Unitization();
    if (dir_unit_res->IsException())
    {
        Log::Error(LOG_NAME, "Failed to unitization direction caused by:");
        dir_unit_res->Print();
    }
    _direction = direction;

    auto upp_set_res = upper.Set(2, 0);
    auto upp_unit_res = upper.Unitization();
    if (upp_set_res->IsException() || upp_unit_res->IsException())
    {
        Log::Error(LOG_NAME, "Failed to set or unitization upper caused by:");
        upp_set_res->Print();
        upp_unit_res->Print();
    }
    _upper = upper;

    SetAngles();
}

void Camera::SetAngles()
{
    _beta = asin(**_direction[1]); // beta (-PI/2, PI/2)
    _alpha = asin((**_direction[0]) / cos(_beta)); // alpha (-PI, PI)
    if(**_direction[2] > 0)
    {
        if(_alpha > 0) 
            _alpha = M_PI - _alpha;
        else
            _alpha = -M_PI - _alpha;
    }
    _gamma = asin(**_upper[0]); // gamma (-PI, PI)
    if(**_upper[1] < 0)
    {
        if(_gamma > 0)
            _gamma = M_PI - _gamma;
        else
            _gamma = -M_PI - _gamma;
    }
}

Camera::Camera(Camera const &camera) : _pvertices(camera._pvertices), _alpha(camera._alpha), _beta(camera._beta), _gamma(camera._gamma), _nearer_dest(camera._nearer_dest), _further_dest(camera._further_dest), _screen_width(camera._screen_width), _screen_height(camera._screen_height)
{
    _location = *camera._location.Copy();
    _direction = *camera._direction.Copy();
}

void Camera::SetVertices(std::vector<Maths::Vectors::Vector> &vertices, std::vector<Maths::Vectors::Vector> &vertices_transform)
{
    _pvertices = &vertices;
    _pvertices_transform = &vertices_transform;
}

DefaultResult Camera::Transform(bool is_print)
{
    CHECK_MEMORY_FOR_DEFAULT_RESULT(_pvertices, LOG_NAME, CAMERA_CODE_NULL_POINTER_PVERTICES)

    SetAngles();

    Log::Trace(LOG_NAME, "vertices count: %d", _pvertices->size());
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
    auto lx = **_location[0];
    auto ly = **_location[1];
    auto lz = **_location[2];

    Log::Trace(LOG_NAME, "a5*4*3 * a2*1 * v:");

    SMatrix a1 = 
    {
        1, 0, 0, -lx,
        0, 1, 0, -ly,
        0, 0, 1, -lz,
        0, 0, 0, 1  
    };

    SMatrix a2 = 
    {
        cos_a, -sin_a_sin_b, sin_a_cos_b, 0,
        0, cos_b, sin_b, 0,
        -sin_a, -cos_a_sin_b, cos_a_cos_b, 0,
        0, 0, 0, 1
    };

    SMatrix a2_1 = 
    {
        1, 0, 0, 0,
        0, cos_b, sin_b, 0,
        0, -sin_b, cos_b, 0,
        0, 0, 0, 1
    };

    SMatrix a2_2 = 
    {
        cos_a, 0, sin_a, 0,
        0, 1, 0, 0,
        -sin_a, 0, cos_a, 0,
        0, 0, 0, 1
    };

    SMatrix a2_3 = 
    {
        cos_b, sin_b, 0, 0,
        -sin_b, cos_b, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    };
    
    SMatrix a21 = 
    {
        cos_a, -sin_a_sin_b, sin_a_cos_b, -lx * cos_a + ly * sin_a_sin_b - lz * sin_a_cos_b,
        0, cos_b, sin_b, -ly * cos_b - lz * sin_b,
        -sin_a, -cos_a_sin_b, cos_a_cos_b, lx * sin_a + ly * cos_a_sin_b - lz * cos_a_cos_b,
        0, 0, 0, 1
    };

    SMatrix a543 = 
    {
        (double)_screen_width * _nearer_dest * cos_g / 2, -(double)_screen_width * _nearer_dest * sin_g / 2, (double)_screen_width / 2, 0,
        -(double)_screen_height * _nearer_dest * sin_g / 2, -(double)_screen_height * _nearer_dest * cos_g / 2, (double)_screen_height / 2, 0,
        0, 0, _nearer_dest + _further_dest, -_nearer_dest * _further_dest,
        0, 0, 1, 0
    };

    // copy vertices_transform from vertices(origin) and transform it.

    for(auto i = 0; i != _pvertices_transform->size(); i++)
    {
        _pvertices_transform->at(i).CopyFrom(_pvertices->at(i));

        Log::Trace(LOG_NAME, "Start a vertice transform...");
        
        _pvertices_transform->at(i).PrintVector(is_print);
        // a21 * _pvertices_transform->at(i);
        a1 * _pvertices_transform->at(i);
        _pvertices_transform->at(i).PrintVector(is_print);
        a2_1 * _pvertices_transform->at(i);
        _pvertices_transform->at(i).PrintVector(is_print);
        a2_2 * _pvertices_transform->at(i);
        _pvertices_transform->at(i).PrintVector(is_print);
        a2_3 * _pvertices_transform->at(i);
        _pvertices_transform->at(i).PrintVector(is_print);

        a543 * _pvertices_transform->at(i);
        _pvertices_transform->at(i).PrintVector(is_print);
        _pvertices_transform->at(i) *= (1 / **(_pvertices_transform->at(i)[3]));
        _pvertices_transform->at(i).PrintVector(is_print);
    }
    //
    return DEFAULT_RESULT;
}