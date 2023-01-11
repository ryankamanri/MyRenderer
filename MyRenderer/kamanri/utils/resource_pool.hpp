#pragma once

#include <vector>
#include <mutex>
#include <condition_variable>

#include "logs.hpp"
#include "string.hpp"
#include "result.hpp"
#include "memory.hpp"

namespace Kamanri
{
    namespace Utils
    {

        namespace __ResourcePool
        {
            constexpr const char *LOG_NAME = STR(Kamanri::Utils::ResourcePool);
        } // namespace __ResourcePool
        

        
        template <class T, byte size>
        class ResourcePool
        {
        private:
            T _pool[size];
            byte _current_count = size;
            std::mutex _mutex;
            std::condition_variable _cv;

            bool _resource_bitmap[size] = {0};
            byte _current_index = 0;

            void _Wait()
            {
                std::unique_lock<std::mutex> lock(_mutex);
                if(--_current_count < 0)
                {
                    _cv.wait(lock, [this](){ return _current_count >= 0; });
                }
            }

            void _Signal()
            {
                std::lock_guard<std::mutex> unlock(_mutex);
                if(++_current_count <= 0)
                {
                    _cv.notify_one();
                }
            }

        public:
            ResourcePool(Result<T> (*set_origin_item)())
            {
                for(int i = 0; i < size; i++)
                {
                    Result<T> origin_item_result = set_origin_item();
                    if(origin_item_result.IsException())
                    {
                        Log::Error(__ResourcePool::LOG_NAME, "An error occured while init the pool item");
                        origin_item_result.Print();
                        return;
                    }
                    _pool[i] = *origin_item_result;
                    _resource_bitmap[i] = true;
                }
            }

            class AllocateItem;
            void Free(AllocateItem item);

            class AllocateItem
            {
                private:
                    byte _index;
                public:
                    T& data;
                    AllocateItem(byte index, T& i): _index(index), data(i) {}
                    friend void ResourcePool::Free(AllocateItem item);
            };

            AllocateItem Allocate()
            {
                _Wait();
                while(!_resource_bitmap[(++_current_index) %= size]);
                _resource_bitmap[_current_index] = false;
                // now get the free resource
                return AllocateItem(_current_index, _pool[_current_index]);
            }


        };

        template<class T, byte size>
        void ResourcePool<T, size>::Free(AllocateItem item)
        {
            _resource_bitmap[item._index] = true;
            _Signal();
        }

    } // namespace Utils
    
} // namespace Kamanri


