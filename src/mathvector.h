#ifndef MATHVECTOR_H
#define MATHVECTOR_H

#include <memory>
#include <algorithm>
#include <vector>
#include <map>
#include <set>
#include <iterator>
#include <exception>
#include <math.h>


#include <iostream>

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
				std::map<size_t, T> data;
				std::set<size_t> not_nulls;

			public:

				struct predicate
				{
					bool operator()(std::pair<size_t, T> first, std::pair<size_t, T> second)
					{
						return std::abs(first.second) > std::abs(second.second);
					};
				};

				T getElement(size_t position);

				typedef typename std::map<size_t, T>::iterator fast_iterator;
				typedef typename std::map<size_t, T>::const_iterator const_fast_iterator;

				struct iterator
				{
				private:

					size_t position;

				public:

					iterator(size_t _position) : position(_position) {}
					T& getElem() const { return *MathVector::getElement(this->position); }
					void setElement(T& element) { MathVector::insert(element, position); return; }
					T* operator++() { return ++position; }
					T* operator--() { return --position; }
					bool operator==(const iterator& other) const { return position == other.position; }
					bool operator!=(const iterator& other) const { return !(*this == other); }
				};

				friend iterator;
				friend Measures::MeasureFunction < T > ;
				friend VectorNorm::MathVectorNorm < T > ;
				friend VectorAlgorithm::MatrixSolver::MathMatrixSolver < T > ;

				MathVector<T>();
				MathVector<T>(size_t _size, T _default_value);
				MathVector<T>(const MathVector<T>& other);
				MathVector<T>(const vector<T>& other);
				MathVector<T>(const std::map<size_t, T>& other);
				MathVector<T>(const std::map<size_t, T>& other, const std::set<size_t>& not_nulls);

				void setValues(const vector<T>& other);
				void setValues(size_t _size, T _default_value);
                T update(T factor, const MathVector<T>& values);

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
					return data.begin();
				}

			    fast_iterator fast_end()
				{
					return data.end();
				}

				const_fast_iterator fast_begin() const
				{
					return typename std::map<size_t, T>::const_iterator(data.begin());
				}

				const_fast_iterator fast_end() const
				{
					return typename std::map<size_t, T>::const_iterator(data.end());
				}

				iterator begin()
				{
					return iterator(0);
				}

				iterator end()
				{
					return iterator(this->size);
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

			template<typename T> MathVector<T>::MathVector(const std::map<size_t, T>& other, const std::set<size_t>& not_nulls)
			{
				this->data = other;
				this->not_nulls = not_nulls;
			}

			template<typename T> MathVector<T>::MathVector(const std::map<size_t, T>& other)
			{
				this->data = other;

				fast_iterator begin = this->data.begin();
				fast_iterator end = this->data.end();


				for (; begin != end; ++begin)
				{
					this->not_nulls.insert(begin->first);
				}

                typename std::map<size_t, T>::iterator last = this->data.end();

				last--;

				this->size = last->first + 1;
			}

			template<typename T> MathVector<T>::MathVector(const vector<T>& other)
			{
				this->size = other.size();

				for (size_t index = 0; index < other.size(); ++index)
				{
					if (other.at(index) != 0)
					{
						this->not_nulls.insert(index);
						this->data[index] = other.at(index);
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
						this->data[index] = _default_value;
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
						this->data[index] = other.at(index);
					}
				}

			}

			template<typename T> T MathVector<T>::getElement(size_t position)
			{
				if (this->not_nulls.find(position) == this->not_nulls.end())
				{
					return 0;
				}
				else
				{
					return data.find(position)->second;
				}
			}

			template<typename T> T MathVector<T>::getMaximalElement()
			{
				fast_iterator position = std::max_element(this->data.begin(), this->data.end(), predicate());

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

					this->data[this->size - 1] = element;
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
						typename std::map<size_t, T>::iterator it = this->data.end();
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
						this->data[index] = _value;
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
					typename std::map<size_t, T>::iterator pos_it = this->data.find(position);
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
				std::map<size_t, int>::iterator begin = this->data.begin();

				return begin->first;
			}

			template<typename T> size_t  MathVector<T>::last_not_null()
			{
				std::map<size_t, int>::iterator end = this->data.end();

				return end->first;
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

				typename std::map<size_t, T>::iterator begin = this->data.begin();
				typename std::map<size_t, T>::iterator end = this->data.end();

				for (; begin != end; ++begin)
				{
					converted->at(begin->first) = begin->second;
				}

				return *converted;
			}


            template<typename T> T MathVector<T>:: update(T factor, const MathVector<T>& other)
            {
                T difference = 0.0;
		        MathVector<T>::const_fast_iterator it = other.fast_begin();
		        MathVector<T>::const_fast_iterator end = other.fast_end();

		        for (; it != end; ++it)
		        {
                    T value = this->getElement(it->first);
			        T new_value = value + factor * it->second;
			        difference += pow(abs(new_value - value), 2.);
			        this->insert(new_value, it->first);
		        }

                return difference;
            }


			template<typename T> T  MathVector<T>::operator*(const MathVector<T>& other)
			{
				T result = 0;

				float thisLoadFactor = (float)this->not_nulls.size() / (float)this->size;
				float otherLoadFactor = (float)other.not_nulls.size() / (float)other.size;

				if (thisLoadFactor <= 0.5 || otherLoadFactor <= 0.5)
				{
					if (this->not_nulls.size() > other.not_nulls.size())
					{
						MathVector<T>::const_fast_iterator it = other.fast_begin();

						for (; it != other.fast_end(); ++it)
						{
							if (this->not_nulls.find(it->first)
								!= this->not_nulls.end())
							{
								result += it->second * this->data.at(it->first);
							}
						}
					}
					else
					{
						MathVector<T>::fast_iterator it = this->fast_begin();

						for (; it != this->fast_end(); ++it)
						{
							if (other.not_nulls.find(it->first)
								!= other.not_nulls.end())
							{
								result += it->second * other.data.at(it->first);
							}
						}
					}
				}
				else
				{
					fast_iterator firstBegin = this->fast_begin();
					const_fast_iterator secondBegin = other.fast_begin();

					while (firstBegin != this->fast_end() || secondBegin != other.fast_end())
					{

						if (firstBegin == this->fast_end())
						{
							secondBegin++;
						}
						else if (secondBegin == other.fast_end())
						{
							firstBegin++;
						}
						else if (firstBegin->first == secondBegin->first)
						{
							result += firstBegin->second * secondBegin->second;

							firstBegin++;
							secondBegin++;
						}
						else if (firstBegin->first > secondBegin->first)
						{
							secondBegin++;
						}
						else if (firstBegin->first < secondBegin->first)
						{
							firstBegin++;
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
						begin->second *= value;
					}
				}

				return result;
			}

			template<typename T>MathVector<T> MathVector<T>::operator/(const T& value)
			{
				MathVector<T> result(*this);

				if (value == 0)
				{
					throw std::exception();
				}
				else
				{
					fast_iterator begin = result.fast_begin();
					fast_iterator end = result.fast_end();

					for (; begin != end; ++begin)
					{
						*begin /= value;
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
						size_t newValue = dataIterator->second + value;

						if (value == 0)
						{
							result.not_nulls.erase(result.not_nulls.find(dataIterator->first));
							result.data.erase(dataIterator);
							dataIterator++;
						}
						else
						{
							result.not_nulls.insert(dataIterator->first);
							result.data[dataIterator->first] = newValue;
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

						if (value == 0)
						{
							result.not_nulls.erase(result.not_nulls.find(dataIterator->first));
							result.data.erase(dataIterator);
							dataIterator++;
						}
						else
						{
							result.not_nulls.insert(dataIterator->first);
							result.data[dataIterator->first] = newValue;
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
						begin->second *= value;
					}
				}

				return *this;
			}

			template<typename T> MathVector<T>& MathVector<T>::operator /= (const T& value)
			{
				if (value == 0)
				{
					throw std::exception();
				}
				else
				{
					fast_iterator begin = this->fast_begin();
					fast_iterator end = this->fast_end();

					for (; begin != end; ++begin)
					{
						begin->second /= value;
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
							this->not_nulls.erase(this->not_nulls.find(dataIterator->first));
							this->data.erase(dataIterator);
							dataIterator++;
						}
						else
						{
							this->not_nulls.insert(dataIterator->first);
							this->data[dataIterator->first] = newValue;
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
							this->not_nulls.erase(this->not_nulls.find(dataIterator->first));
							this->data.erase(dataIterator);
							dataIterator++;
						}
						else
						{
							this->not_nulls.insert(dataIterator->first);
							this->data[dataIterator->first] = newValue;
						}
					}

				}

				return *this;
			}

			template<typename T> MathVector<T>  MathVector<T>::operator+(const MathVector<T>& other)
			{
				float thisLoadFactor = (float)this->not_nulls.size() / (float)this->size;
				float otherLoadFactor = (float)other.not_nulls.size() / (float)other.size;
				if (thisLoadFactor <= 0.5 || otherLoadFactor <= 0.5)
				{
					if (this->not_nulls.size() > other.not_nulls.size())
					{
						MathVector<T> result(*this);
						MathVector<T>::const_fast_iterator it = other.fast_begin();

						for (; it != other.fast_end(); ++it)
						{
						    T value = it->second + result.getElement(it->first);
							result.insert(value, it->first);
						}

						return result;
					}
					else
					{
					    MathVector<T> result(other);
						MathVector<T>::fast_iterator it = this->fast_begin();

						for (; it != this->fast_end(); ++it)
						{
							T value = it->second + result.getElement(it->first);
							result.insert(value, it->first);
						}

						return result;
					}
				}
				else
				{
					MathVector<T> result(this->size, 0);
					fast_iterator firstBegin = this->fast_begin();
					const_fast_iterator secondBegin = other.fast_begin();

					while (firstBegin != this->fast_end() || secondBegin != other.fast_end())
					{
						if (firstBegin == this->fast_end())
						{
							T newvalue = secondBegin->second;
							result.insert(newvalue, secondBegin->first);
							secondBegin++;
						}
						else if (secondBegin == other.fast_end())
						{
							T newvalue = firstBegin->second;
							result.insert(newvalue, firstBegin->first);
							firstBegin++;
						}
						else if (firstBegin->first == secondBegin->first)
						{
							T newvalue = firstBegin->second + secondBegin->second;
							result.insert(newvalue, firstBegin->first);
							firstBegin++;
							secondBegin++;
						}
						else if (firstBegin->first > secondBegin->first)
						{
							T newvalue = secondBegin->second;
							result.insert(newvalue, secondBegin->first);
							secondBegin++;
						}
						else if (firstBegin->first < secondBegin->first)
						{
							T newvalue = firstBegin->second;
							result.insert(newvalue, firstBegin->first);
							firstBegin++;
						}
					}

					return result;
				}
			}

			template<typename T>MathVector<T> MathVector<T>::operator-(const MathVector<T>& other)
			{
				MathVector<T> result(this->size, 0.0);
				float thisLoadFactor = (float)this->not_nulls.size() / (float)this->size;
				float otherLoadFactor = (float)other.not_nulls.size() / (float)other.size;

				if (thisLoadFactor <= 0.5 || otherLoadFactor <= 0.5)
				{
					if (this->not_nulls.size() > other.not_nulls.size())
					{
						MathVector<T>::const_fast_iterator it = other.fast_begin();

						for (; it != other.fast_end(); ++it)
						{
							T value = this->data.at(it->first) - it->second;
							result.insert(value, it->first);
						}
					}
					else
					{
						MathVector<T>::fast_iterator it = this->fast_begin();

						for (; it != this->fast_end(); ++it)
						{
							T value = it->second - other.data.at(it->first);
							result.insert(value, it->first);
						}
					}
				}
				else
				{
					fast_iterator firstBegin = this->fast_begin();
					const_fast_iterator secondBegin = other.fast_begin();

					while (firstBegin != this->fast_end() || secondBegin != other.fast_end())
					{
						if (firstBegin == this->fast_end())
						{
							T newvalue = (-1.0) * secondBegin->second;

							if (newvalue != 0.0)
							{
								result.insert(newvalue, secondBegin->first);
							}
							secondBegin++;
						}
						else if (secondBegin == other.fast_end())
						{
							T newvalue = firstBegin->second;

							if (newvalue != 0.0)
							{
								result.insert(newvalue, firstBegin->first);
							}
							firstBegin++;
						}
						else if (firstBegin->first == secondBegin->first)
						{
							T newvalue = firstBegin->second - secondBegin->second;

							if (newvalue != 0.0)
							{
								result.insert(newvalue, firstBegin->first);
							}

							firstBegin++;
							secondBegin++;
						}
						else if (firstBegin->first > secondBegin->first)
						{
							T newvalue = (-1.0) * secondBegin->second;

							if (newvalue != 0.0)
							{
								result.insert(newvalue, secondBegin->first);
							}

							secondBegin++;
						}
						else if (firstBegin->first < secondBegin->first)
						{
							T newvalue = firstBegin->second;

							if (newvalue != 0.0)
							{
								result.insert(newvalue, firstBegin->first);
							}

							firstBegin++;
						}
					}
				}
				return result;
			}

			template<typename T> MathVector<T>& MathVector<T>::operator+=(const MathVector<T>& other)
			{
				fast_iterator firstBegin = this->fast_begin();
				const_fast_iterator secondBegin = other.fast_begin();

				std::map<size_t, T> newValues;
				std::set<size_t> newNotNulls;

				while (firstBegin != this->fast_end() || secondBegin != other.fast_end())
				{
					if (firstBegin == this->fast_end())
					{
						T newvalue = secondBegin->second;

						newNotNulls.insert(secondBegin->first);
						newValues[secondBegin->first] = newvalue;

						secondBegin++;
					}
					else if (secondBegin == other.fast_end())
					{
						T newvalue = firstBegin->second;

						newNotNulls.insert(firstBegin->first);
						newValues[firstBegin->first] = newvalue;

						firstBegin++;
					}
					else if (firstBegin->first == secondBegin->first)
					{
						T newvalue = firstBegin->second + secondBegin->second;

						if (newvalue != 0)
						{
							newNotNulls.insert(firstBegin->first);
							newValues[firstBegin->first] = newvalue;
						}

						firstBegin++;
						secondBegin++;
					}
					else if (firstBegin->first > secondBegin->first)
					{
						T newvalue = secondBegin->second;

						newNotNulls.insert(secondBegin->first);
						newValues[secondBegin->first] = newvalue;

						secondBegin++;
					}
					else if (firstBegin->first < secondBegin->first)
					{
						T newvalue = firstBegin->second;

						newNotNulls.insert(firstBegin->first);
						newValues[firstBegin->first] = newvalue;

						firstBegin++;
					}

				}

				this->not_nulls = newNotNulls;
				this->data = newValues;

				return *this;
			}

			template<typename T> MathVector<T>& MathVector<T>::operator-=(const MathVector<T>& other)
			{

				fast_iterator firstBegin = this->fast_begin();
				const_fast_iterator secondBegin = other.fast_begin();

				std::map<size_t, T> newValues;
				std::set<size_t> newNotNulls;

				while (firstBegin != this->fast_end() || secondBegin != other.fast_end())
				{
					if (firstBegin == this->fast_end())
					{
						T newvalue = -secondBegin->second;

						newNotNulls.insert(secondBegin->first);
						newValues[secondBegin->first] = newvalue;

						secondBegin++;
					}
					else if (secondBegin == other.fast_end())
					{
						T newvalue = -firstBegin->second;

						newNotNulls.insert(firstBegin->first);
						newValues[firstBegin->first] = newvalue;

						firstBegin++;
					}
					else
						if (firstBegin->first == secondBegin->first)
						{
						T newvalue = firstBegin->second - secondBegin->second;

						if (newvalue != 0)
						{
							newNotNulls.insert(firstBegin->first);
							newValues[firstBegin->first] = newvalue;
						}

						firstBegin++;
						secondBegin++;
						}
						else if (firstBegin->first > secondBegin->first)
						{
							T newvalue = -secondBegin->second;

							newNotNulls.insert(secondBegin->first);
							newValues[secondBegin->first] = newvalue;

							secondBegin++;
						}
						else if (firstBegin->first < secondBegin->first)
						{
							T newvalue = -firstBegin->second;

							newNotNulls.insert(firstBegin->first);
							newValues[firstBegin->first] = newvalue;

							firstBegin++;
						}
				}

				this->not_nulls = newNotNulls;
				this->data = newValues;

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
