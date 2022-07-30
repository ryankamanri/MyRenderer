#include <math.h>
#include "../../utils/logs.h"
#include "../../renderer/cameras.h"

using namespace Kamanri::Renderer::Cameras;
using namespace Kamanri::Utils::Logs;
using namespace Kamanri::Maths::Vectors;

constexpr const char *LOG_NAME = "Kamanri::Renderer::Cameras";


Camera::Camera(Vector location, Vector direction, Vector upper)
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

    _beta = asin(**_direction[1]);
    _alpha = asin((**_direction[0]) / cos(_beta));
    _gamma = asin(**_upper[0]);
}

Camera::Camera(Camera const& camera): _alpha(camera._alpha), _beta(camera._beta), _gamma(camera._gamma)
{
    _location = *camera._location.Copy();
    _direction = *camera._direction.Copy();
}