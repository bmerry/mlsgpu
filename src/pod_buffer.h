/*
 * mlsgpu: surface reconstruction from point clouds
 * Copyright (C) 2013  University of Cape Town
 *
 * This file is part of mlsgpu.
 *
 * mlsgpu is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * @file
 *
 * Vector-like class that only supports explicit resize, and does not default-initialize.
 */

#ifndef POD_BUFFER_H
#define POD_BUFFER_H

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <memory>
#include <cstdlib>
#include <boost/noncopyable.hpp>
#include <string>
#include "errors.h"

/**
 * Vector-like class that only supports explicit resize, and does not
 * default-initialize. It is only suitable for storing POD types.
 *
 * It also does not store an explicit end. The user is supposed to know
 * how much they want to store in it. Thus, it also does not have methods
 * like @c end or @c push_back.
 */
template<typename T, typename Allocator = std::allocator<T> >
class PODBuffer : public boost::noncopyable
{
public:
    typedef typename Allocator::size_type size_type;
    typedef typename Allocator::difference_type difference_type;

    typedef T value_type;
    typedef T &reference;
    typedef const T &const_reference;

    reference operator[](size_type index)
    {
        return data_[index];
    }

    const_reference operator[](size_type index) const
    {
        return data_[index];
    }

    T *data() { return data_; }
    const T *data() const { return data_; }

    size_type capacity() const { return capacity_; }

    void reserve(size_type size, bool preserve = true)
    {
        MLSGPU_ASSERT(size <= allocator_.max_size(), std::invalid_argument);
        if (size > capacity_)
        {
            size_type new_capacity = capacity_ * 2;
            if (new_capacity < capacity_ || size > new_capacity || new_capacity > allocator_.max_size())
            {
                // The < check catches overflow
                new_capacity = size;
            }
            if (data_ == NULL)
            {
                data_ = allocator_.allocate(new_capacity);
            }
            else if (preserve)
            {
                T *new_data = allocator_.allocate(new_capacity);
                std::memcpy(new_data, data_, capacity_ * sizeof(T));
                allocator_.deallocate(data_, capacity_);
                data_ = new_data;
            }
            else
            {
                allocator_.deallocate(data_, capacity_);
                data_ = allocator_.allocate(new_capacity);
            }
            capacity_ = new_capacity;
        }
    }

    explicit PODBuffer(Allocator allocator = Allocator())
        : allocator_(allocator), data_(NULL), capacity_(0) {}

    explicit PODBuffer(size_type capacity, Allocator allocator = Allocator())
        : allocator_(allocator), data_(NULL), capacity_(0)
    {
        reserve(capacity);
    }

    ~PODBuffer()
    {
        if (data_ != NULL)
            allocator_.deallocate(data_, capacity_);
    }

private:
    Allocator allocator_;
    T *data_;
    size_type capacity_;
};

#endif /* POD_BUFFER_H */
