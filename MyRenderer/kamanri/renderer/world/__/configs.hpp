#pragma once
// #include "triangle3d.hpp"


namespace Kamanri
{
	namespace Renderer
	{
		namespace World
		{
			namespace __
			{
				class Configs
				{
					public:
					bool is_commited = false;
					bool is_shadow_mapping = false;
					bool is_use_cuda = false;
					Configs& operator=(Configs const& other)
					{
						is_commited = other.is_commited;
						is_shadow_mapping = other.is_shadow_mapping;
						is_use_cuda = other.is_use_cuda;
						return *this;
					}

				};
                

			} // namespace __

		} // namespace World

	} // namespace Renderer

} // namespace Kamanri

