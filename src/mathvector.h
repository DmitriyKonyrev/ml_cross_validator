#ifndef MATHVECTOR_H
#define MATHVECTOR_H

#include <memory>
#include <algorithm>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <iterator>
#include <exception>
#include <math.h>


#include <iostream>

#include "math_vector_iterator.h"

using namespace std;

namespace MathCore
{

	namespace AlgebraCore
	{
		namespace Measures
		{
			template <typename T> class MeasureFunction;

		};

		namespace VectorCore
		{
			namespace VectorNorm
			{
				template <typename T> class MathVectorNorm;
			};

			namespace VectorAlgorithm
			{
				namespace MatrixSolver
				{
					template <typename T> class MathMatrixSolver;
				}
			}

			template<typename T> class MathVector
			{
			private:
				size_t size;
				std::unordered_map<size_t, T> data;
				std::set<size_t> not_nulls;

			public:

				struct predicate
				{
					bool operator()(std::pair<size_t, T> first, std::pair<size_t, T> second)
					{
						return std::abs(first.second) > std::abs(second.second);
					};
				};
				
				typedef FastMathVectorIterator<T> fast_iterator;
				typedef ConstFastMathVectorIterator<T> const_fast_iterator;

				friend fast_iterator;
				friend const_fast_iterator;
				friend Measures::MeasureFunction < T > ;
				friend VectorNorm::MathVectorNorm < T > ;
				friend VectorAlgorithm::MatrixSolver::MathMatrixSolver < T > ;

				MathVector<T>();
				MathVector<T>(size_t _size, T _default_value);
				MathVector<T>(const MathVector<T>& other);
				MathVector<T>(const vector<T>& other);
				MathVector<T>(const std::unordered_map<size_t, T>& other);
				MathVector<T>(const std::unordered_map<size_t, T>& other, const std::set<size_t>& not_nulls);

				void setValues(const vector<T>& other);
				void setValues(size_t _size, T _default_value);
                T update(T values_factor, T factor, const MathVector<T>& values);
				T getElement(size_t position) const;

				void push_back(T element);
				T pop_back();
				void insert(T element, size_t position);
				void completeWith(size_t counts, T _value);

				//const T& getElement(size_t position) const;
				T getMaximalElement();
				size_t first_not_null();
				size_t last_not_null();
				const size_t& getSize() const;
				size_t getSizeOfNotNullElements();

				std::vector<T>& to_std_vector();

				T operator*(const MathVector<T>& other);

				MathVector<T> operator*(const T& value);
				MathVector<T> operator/(const T& value);
				MathVector<T> operator+(const T& value);
				MathVector<T> operator-(const T& value);

				MathVector<T>& operator*=(const T& value);
				MathVector<T>& operator/=(const T& value);
				MathVector<T>& operator+=(const T& value);
				MathVector<T>& operator-=(const T& value);

				MathVector<T> operator+(const MathVector<T>& other);
				MathVector<T> operator-(const MathVector<T>& other);

				MathVector<T>& operator+=(const MathVector<T>& other);
				MathVector<T>& operator-=(const MathVector<T>& other);

				bool operator==(const MathVector &other) const;
				bool operator!=(const MathVector &other) const;

				fast_iterator fast_begin()
				{
					return fast_iterator(*this, (size_t)0);
				}

			    fast_iterator fast_end()
				{
					return fast_iterator(*this, (size_t)size);
				}

				const_fast_iterator const_fast_begin() const
				{
					return const_fast_iterator(*this, (size_t)0);
				}

				const_fast_iterator const_fast_end() const
				{
					return const_fast_iterator(*this, (size_t)size);
				}
			};

			template<typename T> MathVector<T>::MathVector()
				: size(0)
			{
			}

			template<typename T> MathVector<T>::MathVector(size_t _size, T _default_value)
				: size(_size)
			{
				if (_default_value != 0)
				{
					for (size_t index = 0; index < _size; ++index)
					{
						this->data[index] = _default_value;
						this->not_nulls.insert(index);
					}

				}
			}

			template<typename T> MathVector<T>::MathVector(const MathVector<T>& other)
				: size(other.size), data(other.data), not_nulls(other.not_nulls)
			{
			}

			template<typename T> MathVector<T>::MathVector(const std::unordered_map<size_t, T>& other, const std::set<size_t>& not_nulls)
			{
				this->data = other;
				this->not_nulls = not_nulls;
			}

			template<typename T> MathVector<T>::MathVector(const std::unordered_map<size_t, T>& other)
			{
				this->data = other;

				typename std::unordered_map<size_t, T>::const_iterator begin = this->data.begin();
				typename std::unordered_map<size_t, T>::const_iterator end = this->data.end();

				for (; begin != end; ++begin)
				{
					this->not_nulls.insert(begin->first);
				}

                typename std::set<size_t, T>::iterator last = this->not_nulls.end();
				last--;
				this->size = *last + 1;
			}

			template<typename T> MathVector<T>::MathVector(const vector<T>& other)
			{
				this->size = other.size();

				for (size_t index = 0; index < other.size(); ++index)
				{
					if (other.at(index) != 0)
					{
						this->not_nulls.insert(index);
						this->data.insert(std::make_pair(index, other.at(index)));
					}
				}
			}


			template<typename T> void MathVector<T>::setValues(size_t _size, T _default_value)
			{
				this->size = _size;

				this->not_nulls.clear();
				this->data.clear();

				if (_default_value != 0)
				{
					for (size_t index = 0; index < _size; ++index)
					{
						this->data.insert(std::make_pair(index,  _default_value));
						this->not_nulls.insert(index);
					}
				}
			}

			template<typename T> void MathVector<T>::setValues(const vector<T>& other)
			{
				this->size = other.size();

				this->not_nulls.clear();
				this->data.clear();

				for (size_t index = 0; index < other.size(); ++index)
				{
					if (other.at(index) != 0)
					{
						this->not_nulls.insert(index);
						this->data.insert(std::make_pair(index, other.at(index)));
					}
				}
			}

			template<typename T> T MathVector<T>::getElement(size_t position) const
			{
				typename std::unordered_map<size_t, T>::const_iterator elem_it = this->data.find(position);
				if (elem_it == this->data.end())
				{
					return 0;
				}
				else
				{
					return elem_it->second;
				}
			}

			template<typename T> T MathVector<T>::getMaximalElement()
			{
				typename std::unordered_map<size_t, T>::iterator position = std::max_element(this->data.begin(), this->data.end(), predicate());

				if (position == this->data.end() ||
					position->second < 0)
							return 0.;

				return position->second;
			}

			template<typename T> void  MathVector<T>::push_back(T element)
			{
				this->size++;

				if (element != 0)
				{
					this->not_nulls.insert(this->size - 1);

					this->data.insert(std::make_pair(this->size - 1, element));
				}

				return;
			}

			template<typename T> T MathVector<T>::pop_back()
			{
				T poppedElement;

				if (this->size != 0)
				{
					this->size--;

					if (this->not_nulls.find(this->size)
						!= this->not_nulls.end())
					{
						std::set<size_t>::iterator backIt
							= this->not_nulls.end();
						backIt--;
						this->not_nulls.erase(backIt);
						poppedElement = this->data.at(this->size);
						typename std::unordered_map<size_t, T>::iterator it = this->data.end();
						this->data.erase(--it);
					}
					else
					{
						poppedElement = 0;
					}
				}

				return poppedElement;
			}

			template<typename T> void  MathVector<T>::completeWith(size_t counts, T _value)
			{
				size_t begin = this->size;
				this->size += counts;

				if (_value != 0)
				{
					for (size_t index = begin; index < this->size; ++index)
					{
						this->not_nulls.insert(index);
						this->data.insert(std::make_pair(index,  _value));
					}
				}
			}

			template<typename T> void  MathVector<T>::insert(T element, size_t position)
			{
				if (position > this->size)
				{
					this->size = position + 1;
				}

				if (element != 0)
				{
					typename std::unordered_map<size_t, T>::iterator pos_it = this->data.find(position);
					if (pos_it != this->data.end())
						pos_it->second = element;
					else
                    {
					    this->not_nulls.insert(position);
						this->data.insert(std::make_pair(position, element));
                    }
				}

				return;
			}

			template<typename T> size_t  MathVector<T>::first_not_null()
			{
				std::set<size_t>::iterator begin = this->not_nulls.begin();

				return this->data.find(*begin)->second;
			}

			template<typename T> size_t  MathVector<T>::last_not_null()
			{
				std::set<size_t>::iterator end = this->not_nulls.end();

				return this->data.find(*end)->second;
			}

			template<typename T> const size_t&  MathVector<T>::getSize() const
			{
				return size;
			}

			template<typename T> size_t  MathVector<T>::getSizeOfNotNullElements()
			{
				return this->data.size();
			}

			template<typename T> std::vector<T>& MathVector<T>::to_std_vector()
			{
				std::vector<T>* converted = new std::vector<T>(this->size, 0);

				typename std::set<size_t>::iterator begin = this->not_nulls.begin();
				typename std::set<size_t>::iterator end = this->not_nulls.end();

				for (; begin != end; ++begin)
				{
					converted->at(*begin) = this->data.find(*begin)->second;
				}

				return *converted;
			}


            template<typename T> T MathVector<T>::update(T values_factor, T factor, const MathVector<T>& other)
            {
                T difference = 0.0;
		        MathVector<T>::const_fast_iterator it = other.const_fast_begin();
		        MathVector<T>::const_fast_iterator end = other.const_fast_end();

		        for (; it != end; ++it)
		        {
                    T value = this->getElement(it.index());
			        T new_value = values_factor * value + factor * it.getElem();
			        difference += pow(abs(new_value - value), 2.);
			        this->insert(new_value, it.index());
		        }

                return difference;
            }


			template<typename T> T  MathVector<T>::operator*(const MathVector<T>& other)
			{
				T result = 0;

				double thisLoadFactor = (double)this->not_nulls.size() / (double)this->size;
				double otherLoadFactor = (double)other.not_nulls.size() / (double)other.size;

				if (thisLoadFactor <= 0.5 || otherLoadFactor <= 0.5)
				{
					if (this->not_nulls.size() > other.not_nulls.size())
					{
						MathVector<T>::const_fast_iterator it  = other.const_fast_begin();
						MathVector<T>::const_fast_iterator end = other.const_fast_end();

						for (; it != end; ++it)
						{
							if (this->not_nulls.find(it.index())
								!= this->not_nulls.end())
							{
								result += it.getElem() * this->data.at(it.index());
							}
						}
					}
					else
					{
						MathVector<T>::fast_iterator it  = this->fast_begin();
						MathVector<T>::fast_iterator end = this->fast_end(); 
						for (; it != end; ++it)
						{
							if (other.not_nulls.find(it.index())
								!= other.not_nulls.end())
							{
								result += it.getElem() * other.data.at(it.index());
							}
						}
					}
				}
				else
				{
					fast_iterator firstBegin = this->fast_begin();
					fast_iterator firstEnd   = this->fast_end();
					const_fast_iterator secondBegin = other.const_fast_begin();
					const_fast_iterator secondEnd   = other.const_fast_end();

					while (firstBegin != firstEnd || secondBegin != secondEnd)
					{
						if (firstBegin == firstEnd)
						{
							++secondBegin;
						}
						else if (secondBegin == secondEnd)
						{
							++firstBegin;
						}
						else if (firstBegin.index() == secondBegin.index())
						{
							result += firstBegin.getElem() * secondBegin.getElem();

							++firstBegin;
							++secondBegin;
						}
						else if (firstBegin.index() > secondBegin.index())
						{
							++secondBegin;
						}
						else if (firstBegin.index() < secondBegin.index())
						{
							++firstBegin;
						}
					}
				}

				return result;
			}

			template<typename T>MathVector<T> MathVector<T>::operator*(const T& value)
			{
				MathVector<T> result(*this);

				if (value == 0)
				{
					result.data.clear();
					result.not_nulls.clear();
				}
				else
				{
					fast_iterator begin = result.fast_begin();
					fast_iterator end = result.fast_end();

					for (; begin != end; ++begin)
					{
						 T new_value = begin.getElem() * value;
						 begin.setElement(new_value);
					}
				}
				return result;
			}

			template<typename T>MathVector<T> MathVector<T>::operator/(const T& value)
			{
				MathVector<T> result(*this);

				if (value == 0)
				{
					throw std::logic_error("division by zero");
				}
				else
				{
					fast_iterator begin = result.fast_begin();
					fast_iterator end = result.fast_end();

					for (; begin != end; ++begin)
					{
						T new_value = begin.getElem() / value;
						begin.setElement(new_value);
					}
				}
				return result;
			}

			template<typename T>MathVector<T> MathVector<T>::operator+(const T& value)
			{
				MathVector<T> result(*this);

				if (value != 0)
				{
					result.not_nulls.clear();

					fast_iterator dataIterator = result.fast_begin();
					size_t dataSize = result.data.size();

					for (size_t index = 0; index < dataSize; ++index)
					{
						size_t newValue = dataIterator.getElem() + value;
						size_t dataIndex = dataIterator.index();
						if (value == 0)
						{
							result.not_nulls.erase(result.not_nulls.find(dataIndex));
							result.data.erase(result.data.find(dataIndex));
							dataIterator++;
						}
						else
						{
							result.not_nulls.insert(dataIndex);
							result.data[dataIndex] = newValue;
						}
					}
				}
				return result;
			}

			template<typename T>MathVector<T> MathVector<T>::operator-(const T& value)
			{
				MathVector<T> result(*this);

				if (value != 0)
				{
					result.not_nulls.clear();

					fast_iterator dataIterator = result.fast_begin();
					size_t dataSize = result.data.size();

					for (size_t index = 0; index < dataSize; ++index)
					{
						size_t newValue = dataIterator->second - value;
						size_t dataIndex = dataIterator.index();
						if (value == 0)
						{
							result.not_nulls.erase(result.not_nulls.find(dataIndex));
							result.data.erase(result.data.find(dataIndex));
							dataIterator++;
						}
						else
						{
							result.not_nulls.insert(dataIndex);
							result.data[dataIndex] = newValue;
						}
					}
				}

				return result;
			}

			template<typename T> MathVector<T>& MathVector<T>::operator*=(const T& value)
			{

				if (value == 0)
				{
					this->data.clear();
					this->not_nulls.clear();
				}
				else
				{
					fast_iterator begin = this->fast_begin();
					fast_iterator end = this->fast_end();

					for (; begin != end; ++begin)
					{
						T new_value = begin.getElem() * value;
						begin.setElement(new_value);
					}
				}

				return *this;
			}

			template<typename T> MathVector<T>& MathVector<T>::operator /= (const T& value)
			{
				if (value == 0)
				{
					throw std::logic_error("division by zero");
				}
				else
				{
					fast_iterator begin = this->fast_begin();
					fast_iterator end = this->fast_end();

					for (; begin != end; ++begin)
					{
						T new_value = begin.getElem() / value;
						begin.setElement(new_value);
					}
				}

				return *this;
			}

			template<typename T> MathVector<T>& MathVector<T>::operator+=(const T& value)
			{
				if (value != 0)
				{
					this->not_nulls.clear();

					fast_iterator dataIterator = this->fast_begin();
					size_t dataSize = this->data.size();

					for (size_t index = 0; index < dataSize; ++index)
					{
						size_t newValue = dataIterator->second + value;

						if (value == 0)
						{
							this->not_nulls.erase(this->not_nulls.find(dataIterator.index()));
							this->data.erase(this->data.find(dataIterator.index()));
							dataIterator++;
						}
						else
						{
							this->not_nulls.insert(dataIterator.index());
							this->data[dataIterator.index()] = newValue;
						}
					}
				}

				return *this;
			}

			template<typename T> MathVector<T>& MathVector<T>::operator-=(const T& value)
			{
				if (value != 0)
				{
					this->not_nulls.clear();

					fast_iterator dataIterator = this->fast_begin();
					size_t dataSize = this->data.size();

					for (size_t index = 0; index < dataSize; ++index)
					{
						size_t newValue = dataIterator->second - value;

						if (value == 0)
						{
							this->not_nulls.erase(this->not_nulls.find(dataIterator.index()));
							this->data.erase(this->data.find(dataIterator.index()));
							dataIterator++;
						}
						else
						{
							this->not_nulls.insert(dataIterator);
							this->data[dataIterator->first] = newValue;
						}
					}
				}

				return *this;
			}

			template<typename T> MathVector<T>  MathVector<T>::operator+(const MathVector<T>& other)
			{
				double thisLoadFactor = (double)this->not_nulls.size() / (double)this->size;
				double otherLoadFactor = (double)other.not_nulls.size() / (double)other.size;
				if (thisLoadFactor <= 0.5 || otherLoadFactor <= 0.5)
				{
					if (this->not_nulls.size() > other.not_nulls.size())
					{
						MathVector<T> result(*this);
						MathVector<T>::const_fast_iterator it = other.const_fast_begin();
						MathVector<T>::const_fast_iterator end = other.const_fast_end();
						for (; it != end; ++it)
						{
						    T value = it.getElem() + result.getElement(it.index());
							result.insert(value, it.index());
						}

						return result;
					}
					else
					{
					    MathVector<T> result(other);
						MathVector<T>::fast_iterator it  = this->fast_begin();
						MathVector<T>::fast_iterator end = this->fast_end();
						for (; it != end; ++it)
						{
							T value = it.getElem() + result.getElement(it.index());
							result.insert(value, it.index());
						}

						return result;
					}
				}
				else
				{
					MathVector<T> result(this->size, 0);
					fast_iterator firstBegin = this->fast_begin();
					fast_iterator firstEnd   = this->fast_end();
					const_fast_iterator secondBegin = other.const_fast_begin();
					const_fast_iterator secondEnd   = other.const_fast_end();

					while (firstBegin != firstEnd || secondBegin != secondEnd)
					{
						if (firstBegin == firstEnd)
						{
							T newvalue = secondBegin.getElem();
							result.insert(newvalue, secondBegin.index());
							++secondBegin;
						}
						else if (secondBegin == secondEnd)
						{
							T newvalue = firstBegin.getElem();
							result.insert(newvalue, firstBegin.index());
							++firstBegin;
						}
						else if (firstBegin.index() == secondBegin.index())
						{
							T newvalue = firstBegin.getElem() + secondBegin.getElem();
							result.insert(newvalue, firstBegin.index());
							++firstBegin;
							++secondBegin;
						}
						else if (firstBegin.index() > secondBegin.index())
						{
							T newvalue = secondBegin.getElem();
							result.insert(newvalue, secondBegin.index());
							++secondBegin;
						}
						else if (firstBegin.index() < secondBegin.index())
						{
							T newvalue = firstBegin.getElem();
							result.insert(newvalue, firstBegin.index());
							++firstBegin;
						}
					}

					return result;
				}
			}

			template<typename T>MathVector<T> MathVector<T>::operator-(const MathVector<T>& other)
			{
				MathVector<T> result(this->size, 0.0);
				double thisLoadFactor = (double)this->not_nulls.size() / (double)this->size;
				double otherLoadFactor = (double)other.not_nulls.size() / (double)other.size;

				if (thisLoadFactor <= 0.5 || otherLoadFactor <= 0.5)
				{
					if (this->not_nulls.size() > other.not_nulls.size())
					{
						MathVector<T>::const_fast_iterator it = other.const_fast_begin();
						MathVector<T>::const_fast_iterator end = other.const_fast_end();

						for (; it != end; ++it)
						{
							T value = this->getElement(it.index()) - it.getElem();
							result.insert(value, it.index());
						}
					}
					else
					{
						MathVector<T>::fast_iterator it  = this->fast_begin();
						MathVector<T>::fast_iterator end = this->fast_end(); 
						for (; it != end; ++it)
						{
							T value = it.getElem() - other.getElement(it.index());
							result.insert(value, it.index());
						}
					}
				}
				else
				{
					fast_iterator firstBegin = this->fast_begin();
					fast_iterator firstEnd   = this->fast_end();
					const_fast_iterator secondBegin = other.const_fast_begin();
					const_fast_iterator secondEnd   = other.const_fast_end();

					while (firstBegin != firstEnd || secondBegin != secondEnd)
					{
						if (firstBegin == firstEnd)
						{
							T newvalue = (-1.0) * secondBegin.getElem();

							if (newvalue != 0.0)
							{
								result.insert(newvalue, secondBegin.index());
							}
							++secondBegin;
						}
						else if (secondBegin == secondEnd)
						{
							T newvalue = firstBegin.getElem();

							if (newvalue != 0.0)
							{
								result.insert(newvalue, firstBegin.index());
							}
							++firstBegin;
						}
						else if (firstBegin.index() == secondBegin.index())
						{
							T newvalue = firstBegin.getElem() - secondBegin.getElem();

							if (newvalue != 0.0)
							{
								result.insert(newvalue, firstBegin.index());
							}

							++firstBegin;
							++secondBegin;
						}
						else if (firstBegin.index() > secondBegin.index())
						{
							T newvalue = (-1.0) * secondBegin.getElem();

							if (newvalue != 0.0)
							{
								result.insert(newvalue, secondBegin.index());
							}

							++secondBegin;
						}
						else if (firstBegin.index() < secondBegin.index())
						{
							T newvalue = firstBegin.getElem();

							if (newvalue != 0.0)
							{
								result.insert(newvalue, firstBegin.index());
							}

							++firstBegin;
						}
					}
				}
				return result;
			}

			template<typename T> MathVector<T>& MathVector<T>::operator+=(const MathVector<T>& other)
			{
				double thisLoadFactor = (double)this->not_nulls.size() / (double)this->size;
				double otherLoadFactor = (double)other.not_nulls.size() / (double)other.size;

				if (thisLoadFactor <= 0.5 || otherLoadFactor <= 0.5)
				{
					if (this->not_nulls.size() > other.not_nulls.size())
					{
						MathVector<T>::const_fast_iterator it = other.const_fast_begin();
						MathVector<T>::const_fast_iterator end = other.const_fast_end();

						for (; it != end; ++it)
						{
							T value = this->getElement(it.index()) + it.getElem();
							this->insert(value, it.index());
						}
					}
					else
					{
						MathVector<T>::fast_iterator it  = this->fast_begin();
						MathVector<T>::fast_iterator end = this->fast_end(); 
						for (; it != end; ++it)
						{
							T value = it.getElem() + other.getElement(it.index());
							this->insert(value, it.index());
						}
					}
				}
				else
				{
					fast_iterator firstBegin = this->fast_begin();
					fast_iterator firstEnd   = this->fast_end();
					const_fast_iterator secondBegin = other.const_fast_begin();
					const_fast_iterator secondEnd   = other.const_fast_end();

					while (firstBegin != firstEnd || secondBegin != secondEnd)
					{
						if (firstBegin == firstEnd)
						{
							T newvalue = secondBegin.getElem();

							if (newvalue != 0.0)
							{
								this->insert(newvalue, secondBegin.index());
							}
							++secondBegin;
						}
						else if (secondBegin == secondEnd)
						{
							T newvalue = firstBegin.getElem();

							if (newvalue != 0.0)
							{
								this->insert(newvalue, firstBegin.index());
							}
							++firstBegin;
						}
						else if (firstBegin.index() == secondBegin.index())
						{
							T newvalue = firstBegin.getElem() + secondBegin.getElem();

							if (newvalue != 0.0)
							{
								this->insert(newvalue, firstBegin.index());
							}

							++firstBegin;
							++secondBegin;
						}
						else if (firstBegin.index() > secondBegin.index())
						{
							T newvalue = secondBegin.getElem();

							if (newvalue != 0.0)
							{
								this->insert(newvalue, secondBegin.index());
							}

							++secondBegin;
						}
						else if (firstBegin.index() < secondBegin.index())
						{
							T newvalue = firstBegin.getElem();

							if (newvalue != 0.0)
							{
								this->insert(newvalue, firstBegin.index());
							}

							++firstBegin;
						}
					}
				}
				return *this;
			}

			template<typename T> MathVector<T>& MathVector<T>::operator-=(const MathVector<T>& other)
			{
				double thisLoadFactor = (double)this->not_nulls.size() / (double)this->size;
				double otherLoadFactor = (double)other.not_nulls.size() / (double)other.size;

				if (thisLoadFactor <= 0.5 || otherLoadFactor <= 0.5)
				{
					if (this->not_nulls.size() > other.not_nulls.size())
					{
						MathVector<T>::const_fast_iterator it = other.const_fast_begin();
						MathVector<T>::const_fast_iterator end = other.const_fast_end();

						for (; it != end; ++it)
						{
							T value = this->getElement(it.index()) - it.getElem();
							this->insert(value, it.index());
						}
					}
					else
					{
						MathVector<T>::fast_iterator it  = this->fast_begin();
						MathVector<T>::fast_iterator end = this->fast_end(); 
						for (; it != end; ++it)
						{
							T value = it.getElem() - other.getElement(it.index());
							this->insert(value, it.index());
						}
					}
				}
				else
				{
					fast_iterator firstBegin = this->fast_begin();
					fast_iterator firstEnd   = this->fast_end();
					const_fast_iterator secondBegin = other.const_fast_begin();
					const_fast_iterator secondEnd   = other.const_fast_end();

					while (firstBegin != firstEnd || secondBegin != secondEnd)
					{
						if (firstBegin == firstEnd)
						{
							T newvalue = (-1.0) * secondBegin.getElem();

							if (newvalue != 0.0)
							{
								this->insert(newvalue, secondBegin.index());
							}
							++secondBegin;
						}
						else if (secondBegin == secondEnd)
						{
							T newvalue = firstBegin.getElem();

							if (newvalue != 0.0)
							{
								this->insert(newvalue, firstBegin.index());
							}
							++firstBegin;
						}
						else if (firstBegin.index() == secondBegin.index())
						{
							T newvalue = firstBegin.getElem() - secondBegin.getElem();

							if (newvalue != 0.0)
							{
								this->insert(newvalue, firstBegin.index());
							}

							++firstBegin;
							++secondBegin;
						}
						else if (firstBegin.index() > secondBegin.index())
						{
							T newvalue = (-1.0) * secondBegin.getElem();

							if (newvalue != 0.0)
							{
								this->insert(newvalue, secondBegin.index());
							}

							++secondBegin;
						}
						else if (firstBegin.index() < secondBegin.index())
						{
							T newvalue = firstBegin.getElem();

							if (newvalue != 0.0)
							{
								this->insert(newvalue, firstBegin.index());
							}

							++firstBegin;
						}
					}
				}
				return *this;
			}

			template<typename T> bool MathVector<T>::operator==(const MathVector &_other) const
			{
				if (this->size != _other.size())
				{
					return false;
				}
				else
				{
					return this->data == _other.data && this->nulls == _other.nulls;
				}
			}

			template<typename T> bool MathVector<T>::operator!=(const MathVector &_other) const
			{
				if (this->size == _other.size())
				{
					return false;
				}
				else
				{
					return this->data != _other.data || this->nulls != _other.nulls;
				}
			}
		}
	}
};

#endif //MATHVECTOR
