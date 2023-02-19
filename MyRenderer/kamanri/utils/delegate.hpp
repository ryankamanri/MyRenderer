#pragma once
#include <typeinfo>

namespace Kamanri
{
	namespace Utils
	{
		template <class T>
		class Delegate
		{

		public:
			// The Node is a item of Delegate chain.
			class ANode
			{
			private:
				friend void Delegate::Execute(T element);
				friend void Delegate::AddHead(ANode &new_delegate);
				friend void Delegate::AddRear(ANode &new_delegate);

			protected:
				ANode *_next_delegate_node = nullptr;
				virtual void Func(T& element) = 0; // 纯虚函数

			public:
				void InvokeNext(T element)
				{
					if (_next_delegate_node == nullptr)
						return;
					_next_delegate_node->Func(element);
				}
			};

			Delegate()
			{
				_head = _rear = nullptr;
			}

			void Execute(T element)
			{
				if (_head == nullptr)
					return;
				_head->Func(element);
			}
			void AddHead(ANode &new_delegate)
			{
				if (new_delegate != nullptr)
					new_delegate._next_delegate_node = _head;
				_head = &new_delegate;
				if (_rear == nullptr)
					_rear = _head;
			}

			void AddRear(ANode &new_delegate)
			{
				if (_rear != nullptr)
					_rear->_next_delegate_node = &new_delegate;
				_rear = &new_delegate;
				if (_head == nullptr)
					_head = _rear;
			}

		private:
			ANode *_head;
			ANode *_rear;
		};


	} // namespace Utils

} // namespace Kamanri
