#pragma once
#include "kamanri/utils/result.hpp"
#include "kamanri/maths/vector.hpp"
#include "kamanri/utils/memory.hpp"
#include "kamanri/renderer/world/__/resources.hpp"
#include "bling_phong_reflection_model.hpp"
namespace Kamanri
{
	namespace Renderer
	{
		namespace World
		{
			namespace Camera$
			{
				constexpr const int CODE_NULL_POINTER_PVERTICES = 100;
				constexpr const int CODE_INVALID_VECTOR_LENGTH = 200;
				constexpr const int CODE_UNEQUAL_NUM = 300;
				class CameraAttributes
				{
				public:
					Maths::Vector location;
					Maths::Vector direction;
					Maths::Vector upper;
					double nearest_dist;
					double furthest_dist;
					unsigned int screen_width;
					unsigned int screen_height;
					CameraAttributes(Maths::Vector $location, Maths::Vector $direction, Maths::Vector $upper, double $nearer_dest, double $further_dest, unsigned int $screen_width, unsigned int $screen_height):
					location($location), direction($direction), upper($upper), nearest_dist($nearer_dest), furthest_dist($further_dest), screen_width($screen_width), screen_height($screen_height) {}
				};
			}

			class Camera
			{
			private:
				/* data */
				///////////////////////////////
				// references
				__::Resources* _p_resources = nullptr;

				BlingPhongReflectionModel* _p_bpr_model = nullptr;
				/////////////////////////////////

				// need 4d vector
				Maths::Vector _location;
				Maths::Vector _direction;
				// the upward direction only need 3d vector
				Maths::Vector _upward;

				// angles
				/**
				 * @brief alpha angle, between x & -z
				 * 
				 */
				double _alpha;
				/**
				 * @brief beta angle, between y & xOz
				 * 
				 */
				double _beta;
				/**
				 * @brief gamma angle, between y & x
				 * 
				 */
				double _gamma;

				/**
				 * @brief the nearer dest in perspective transfomation
				 * 
				 */
				double _nearest_dist;
				/**
				 * @brief the further dest in perspective transfomation 
				 * 
				 */
				double _furthest_dist;

				unsigned int _screen_width;

				unsigned int _screen_height;

				void SetAngles();

			public:
				Camera();
				Camera(Maths::Vector location, Maths::Vector direction, Maths::Vector upper, double nearest_dist, double furthest_dist, unsigned int screen_width, unsigned int screen_height);
				Camera(Camera&& camera);
				Camera& operator=(Camera&& other);
				void __SetRefs(__::Resources& resources, BlingPhongReflectionModel& bpr_model);
				Utils::DefaultResult Transform();
				/**
				 * @brief Inverse the upper vector when the upper of direction changed.
				 * 
				 * @param last_direction 
				 * @return Utils::DefaultResult 
				 */
				Utils::DefaultResult InverseUpperByDirection(Maths::Vector const& last_direction);

				inline Maths::Vector &Location() { return _location; }
				inline Maths::Vector &Direction() { return _direction; }
				inline Maths::Vector &Upper() { return _upward; }

				inline double NearestDist() const { return _nearest_dist; }
				inline unsigned int ScreenWidth() const { return _screen_width; }
				inline unsigned int ScreenHeight() const { return _screen_height; }
				
				
			};
			
			
		} // namespace World
		
	} // namespace Renderer
	
} // namespace Kamanri
