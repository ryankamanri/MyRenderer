#pragma once
#include <algorithm>

namespace Kamanri
{
    namespace Utils
    {
        namespace Iterator
        {

            class Range
            {
                long _from;
                long _to;

            public:
                // 通过继承自 std::iterator 提供成员 typedef
                class iterator : public std::iterator<
                                     std::input_iterator_tag, // iterator_category
                                     long,                    // value_type
                                     long,                    // difference_type
                                     long *,            // pointer
                                     long                     // reference
                                     >
                {
                    long _from;
                    long _to;
                    long _num;

                public:
                    explicit iterator(long from, long to) : _from(from), _to(to), _num(from) {}
                    iterator &operator++()
                    {
                        _num = _to >= _from ? _num + 1 : _num - 1;
                        return *this;
                    }
                    iterator operator++(int)
                    {
                        iterator retval = *this;
                        ++(*this);
                        return retval;
                    }
                    bool operator==(iterator other) const { return _num == other._num; }
                    bool operator!=(iterator other) const { return !(*this == other); }
                    reference& operator*() { return _num; }

                };
                Range(long from, long to): _from(from), _to(to) {}
                iterator begin() { return iterator(_from, _to); }
                iterator end() { return iterator(_to >= _from ? _to + 1 : _to - 1, _to); }
            };
        } // namespace Iterator

    } // namespace Utils

} // namespace Kamanri
