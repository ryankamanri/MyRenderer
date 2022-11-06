#pragma once
namespace Kamanri
{
    namespace Utils
    {
        namespace Delegate$
        {
            typedef struct Node
            {

                void (*this_delegate)(int, struct Node &);
                struct Node *next_delegate_node;
                void Next(int res)
                {
                    if (next_delegate_node != nullptr)
                        next_delegate_node->this_delegate(res, *next_delegate_node);
                }

            } Node;
        } // namespace Delegate$

        class Delegate
        {
        private:
            Delegate$::Node *_head = nullptr;
            Delegate$::Node *_rear = nullptr;

        public:
            void Execute(int value)
            {
                if (_head == nullptr)
                    return;
                _head->this_delegate(value, *_head);
            }
            void AddHead(Delegate$::Node &new_delegate)
            {
                new_delegate.next_delegate_node = _head;
                _head = &new_delegate;
                if (_rear == nullptr)
                    _rear = _head;
            }

            void AddRear(Delegate$::Node &new_delegate)
            {
                _rear->next_delegate_node = &new_delegate;
                _rear = &new_delegate;
                if (_head == nullptr)
                    _head = _rear;
            }
        };

    } // namespace Utils

} // namespace Kamanri
