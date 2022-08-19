#pragma once

#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>
#include "logs.hpp"

namespace Kamanri
{
    namespace Utils
    {
        namespace Thread
        {
            void Sleep(int millis);

            class ThreadPool
            {

            public:
                ThreadPool(size_t);                                                                            //构造函数
                template <class F, class... Args>                                                              //类模板
                auto EnQueue(F &&f, Args &&...args) -> std::future<typename std::result_of<F(Args...)>::type>; //任务入队
                auto Join() -> void;
                ~ThreadPool();                                                                                 //析构函数

            private:
                std::vector<std::thread> workers;        //线程队列，每个元素为一个Thread对象
                std::queue<std::function<void()>> tasks; //任务队列，每个元素为一个函数对象

                std::mutex queue_mutex;            //互斥量
                std::condition_variable condition; //条件变量
                std::mutex empty_join_mutex;
                std::condition_variable empty_condition; // task为空时的条件变量
                bool stop;                         //停止
            };



            // 添加新的任务到任务队列
            template <class F, class... Args>
            auto ThreadPool::EnQueue(F &&f, Args &&...args)
                -> std::future<typename std::result_of<F(Args...)>::type>
            {
                // 获取函数返回值类型
                using return_type = typename std::result_of<F(Args...)>::type;

                // 创建一个指向任务的只能指针
                auto task = std::make_shared<std::packaged_task<return_type()>>(
                    std::bind(std::forward<F>(f), std::forward<Args>(args)...));

                std::future<return_type> res = task->get_future();
                {
                    std::unique_lock<std::mutex> lock(queue_mutex); //加锁
                    if (stop)
                        throw std::runtime_error("enqueue on stopped ThreadPool");

                    tasks.emplace([task]()
                                  { (*task)(); }); //把任务加入队列
                }                                  //自动解锁
                condition.notify_one();            //通知条件变量，唤醒一个线程
                return res;
            }

            
        }
    }
    
}
