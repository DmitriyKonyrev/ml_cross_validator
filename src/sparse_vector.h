#ifndef SPARSEVECTOR_H
#define SPARSEVECTOR_H

#include <algorithm>
#include <map>
#include <vector>

#include "common_vector.h"

namespace MathCore
{
    namespace AlgebraCore
    {
        template<typename T> class SparseVector
            : public CommonVector<T>
        {
        public:
            typedef typename CommonVector<T>::fast_iterator base_fast_iterator;
            typedef typename CommonVector<T>::const_fast_iterator base_const_fast_iterator;

            typedef typename std::map<size_t, T>::iterator mapped_iterator;
			typedef typename std::map<size_t, T>::const_iterator const_mapped_iterator;

            //--------iterators---------------
            class fast_iterator
                : public CommonVector<T>::fast_iterator
            {
            public:
                friend SparseVector<T>;

                typedef fast_iterator self_type;
                typedef typename CommonVector<T>::fast_iterator* self_pointer;
                typedef T value_type;
				typedef T& reference_type;

            protected:
                fast_iterator(SparseVector& parent, size_t index=0)
                    : m_parent(parent)
                {
                    if (index == 0)
                    {
                        this->m_valueIterator = this->m_parent.m_notNullValues.begin();
                        this->m_index = this->m_valueIterator->first;
                    }
                    else if (index == this->m_parent.m_size)
                    {
                        this->m_valueIterator = this->m_parent.m_notNullValues.end();
                        this->m_index = this->m_parent.m_size;

                    }
                    else
                    {
                        mapped_iterator  value_it = this->m_parent.m_notNullValues.find(index);
                        if (value_it != this->m_parent.m_notNullValues.end())
                        {
                            this->m_valueIterator = value_it;
                        }
                        else
                        {
                            mapped_iterator nearestNotNull;
                            for (nearestNotNull = this->m_parent.m_notNullValues.begin();
                                 nearestNotNull != this->m_parent.m_notNullValues.end();
                                 nearestNotNull++)
                            {
                                if (index >= nearestNotNull->first)
                                    break;
                            }
                        }
                        this->m_index = this->m_valueIterator->first;
                    }
                }

            public:
                self_pointer operator++()
                {
					increment();
                    return this;
                }

                void increment()
                {
                    this->m_valueIterator++;
                    if (this->m_valueIterator != this->m_parent.m_notNullValues.end())
                    {
                        this->m_index = this->m_valueIterator->first;
                    }
                    else
                    {
                        this->m_index = this->m_parent.m_size;
					}
                }

                reference_type operator*()
				{
                    return this->m_valueIterator->second;
                }

                reference_type value()
                {
                    return this->m_valueIterator->second;
                }

            private:
                SparseVector& m_parent;
                mapped_iterator m_valueIterator;
            };

            class const_fast_iterator
                : public CommonVector<T>::const_fast_iterator
            {
            public:
                friend SparseVector<T>;

                typedef const_fast_iterator self_type;
                typedef const typename CommonVector<T>::const_fast_iterator* self_pointer;
                typedef T value_type;

            protected:
                const_fast_iterator(const SparseVector& parent, size_t index=0)
                    : m_parent(parent)
                {
                    if (index == 0)
                    {
                        this->m_valueIterator = this->m_parent.m_notNullValues.begin();
                        this->m_index = this->m_valueIterator->first;
                    }
                    else if (index == this->m_parent.m_size)
                    {
                        this->m_valueIterator = this->m_parent.m_notNullValues.end();
                        this->m_index = this->m_parent.m_size;

                    }
                    else
                    {
                        const_mapped_iterator  value_it = this->m_parent.m_notNullValues.find(index);
                        if (value_it != this->m_parent.m_notNullValues.end())
                        {
                            this->m_valueIterator = value_it;
                        }
                        else
                        {
                            const_mapped_iterator nearestNotNull;
                            for (nearestNotNull = this->m_parent.m_notNullValues.begin();
                                 nearestNotNull != this->m_parent.m_notNullValues.end();
                                 nearestNotNull++)
                            {
                                if (index >= nearestNotNull->first)
                                    break;
                            }
                        }
                        this->m_index = this->m_valueIterator->first;
                    }
                }

            public:
                void increment()
                {
                    this->m_valueIterator++;
                    if (this->m_valueIterator != this->m_parent.m_notNullValues.end())
                    {
                        this->m_index = this->m_valueIterator->first;
                    }
                    else
                    {
                        this->m_index = this->m_parent.m_size;
                    }
                }

                value_type operator*() const
                {
                    return this->m_valueIterator->second;
                }

                value_type value() const
                {
                    return this->m_valueIterator->second;
                }

            private:
                const SparseVector& m_parent;
                const_mapped_iterator m_valueIterator;
            };

        public:
            SparseVector(size_t size, T default_value=0.0)
                : CommonVector<T>(size, default_value)
            {
                if (default_value == 0.0)
                {
                    for (size_t index = 0; index < size; ++index)
                    {
                        this->m_notNullIndexes.insert(index);
                        this->m_notNullValues.insert(std::make_pair(index, default_value));
                    }
                }
            }

            SparseVector(const std::vector<T>& values)
                : CommonVector<T>(values)
            {
                for (size_t index = 0; index < this->m_size; ++index)
                {
                    if (values.at(index) != 0)
                    {
                        this->m_notNullIndexes.insert(index);
                        this->m_notNullValues.insert(std::make_pair(index, values.at(index)));
                    }
                }
            }

            SparseVector(const std::map<size_t, T> mapped_values, size_t full_size)
                : CommonVector<T>(mapped_values, full_size),
                  m_notNullValues(mapped_values)
            {
				const_mapped_iterator it   = mapped_values.begin();
				const_mapped_iterator end  = mapped_values.end();

				for (; it != end; ++it)
					this->m_notNullIndexes.insert(it->first);
            };

			void update(const T& value, size_t index)
			{
				if (index >= this->m_size)
					throw std::length_error("Index out of range");

				else
				{
					m_notNullValues.insert(std::make_pair(index, value));
					m_notNullIndexes.insert(index);
				}
			}

            //------------iteration-operators--------
            T get_value(size_t index)
            {
                mapped_iterator value_it = this->m_notNullValues.find(index);
                if (value_it != this->m_notNullValues.end())
                {
                    return value_it->second;
                }
                else
                {
                    return (T)0;
                }
            }

			size_t get_not_nulls_count()
			{
				return m_notNullIndexes.size();
			}

            bool is_null()
            {
                return this->m_notNullIndexes.size() == 0;
            }
			
			void extend(size_t counts)
			{
				CommonVector<T>::extend(counts);

			}

			void clear()
			{
				m_notNullValues.clear();
				m_notNullIndexes.clear();
				CommonVector<T>::clear();
			}

            T operator[](size_t index)
            {
               return this->get_value(index);
            }

            base_fast_iterator* fast_begin()
            {
                return new fast_iterator(*this, 0);
            }

            base_fast_iterator* fast_end()
            {
                return new fast_iterator(*this, this->m_size);
            }

            base_const_fast_iterator* const_fast_begin() const
            {
                return new const_fast_iterator(*this, 0);
            }

            base_const_fast_iterator* const_fast_end() const
            {
                return new const_fast_iterator(*this, this->m_size);
            }

        private:
            std::map<size_t, T> m_notNullValues;
            std::set<size_t> m_notNullIndexes;
        };
    }
}

#endif//SPARSEVECTOR_H
