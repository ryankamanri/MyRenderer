#pragma once
#ifndef SWIG
#include "kamanri/renderer/world/camera$.hpp"
#endif
namespace Kamanri
{
	namespace Renderer
	{
		namespace World
		{

			class Camera
			{
			private:
				/* data */
				///////////////////////////////
				// references
				Kamanri::Renderer::World::__::Resources* _p_resources = nullptr;

				Kamanri::Renderer::World::BlinnPhongReflectionModel* _p_bpr_model = nullptr;
				/////////////////////////////////

				// need 4d vector
				Kamanri::Maths::Vector _location;
				Kamanri::Maths::Vector _direction;
				// the upward direction only need 3d vector
				Kamanri::Maths::Vector _upward;

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
				Camera(Kamanri::Maths::Vector location, Kamanri::Maths::Vector direction, Kamanri::Maths::Vector upper, double nearest_dist, double furthest_dist, unsigned int screen_width, unsigned int screen_height);
				Camera(Camera&& camera);
				Camera& operator=(Camera const& other);
				Camera& operator=(Camera&& other);
				void __SetRefs(Kamanri::Renderer::World::__::Resources& resources, Kamanri::Renderer::World::BlinnPhongReflectionModel& bpr_model);
				int Transform(bool is_transform_bpr_model = true);
				/**
				 * @brief Inverse the upper vector when the upper of direction changed.
				 * 
				 * @param last_direction 
				 * @return Utils::DefaultResult 
				 */
				int InverseUpperByDirection(Kamanri::Maths::Vector const& last_direction);

				inline Kamanri::Maths::Vector &Location() { return _location; }
				inline Kamanri::Maths::Vector &Direction() { return _direction; }
				inline Kamanri::Maths::Vector &Upward() { return _upward; }
#ifdef __CUDA_RUNTIME_H__ 
				__device__
#endif
				inline double NearestDist() const { return _nearest_dist; }
				inline unsigned int ScreenWidth() const { return _screen_width; }
				inline unsigned int ScreenHeight() const { return _screen_height; }
				
				
			};
			
			
		} // namespace World
		
	} // namespace Renderer
	
} // namespace Kamanri
