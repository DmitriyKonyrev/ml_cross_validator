#ifndef MATHVECTOR_ITERATOR_H
#define MATHVECTOR_ITERATOR_H

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

#include "mathvector.h"

using namespace std;

namespace MathCore
{
	namespace AlgebraCore
	{
	namespace VectorCore
	{
		template<typename T> class MathVector;

		template <typename T> struct FastMathVectorIterator
		{
		private:
			MathVector<T>& parent;
			std::set<size_t>::iterator m_element_it;

		public:
			FastMathVectorIterator(MathVector<T>& _parent, size_t _position) 
			: parent(_parent)
			{
				if (_position == 0)
					m_element_it = parent.not_nulls.begin();
				else if (_position >= parent.size)
					m_element_it = parent.not_nulls.end();
				else
				{
					m_element_it = parent.not_nulls.find(_position);
					if (m_element_it == parent.not_nulls.end())
					{
						std::set<size_t>::iterator it = parent.not_nulls.begin();
						size_t min = parent.size;

						for (; it != parent.not_nulls.end(); ++it)
						{
							size_t distance = abs(*it - _position);
							if (distance <= min)
							{
								m_element_it = it;
								min = distance;
							}
						}
					}
				}
			}

			T getElem() const
			{ 
				return parent.getElement(*m_element_it); 
			}

			void setElement(T& element) 
			{ 
				parent.insert(element, *m_element_it); 
				return; 
			}

			size_t index() const
			{
				return *m_element_it;
			}

			void operator++() { ++m_element_it; }
			void operator--() { --m_element_it; }

			bool operator==(const FastMathVectorIterator& other) const { return m_element_it == other.m_element_it; }
			bool operator!=(const FastMathVectorIterator& other) const { return !(*this == other); }
		};

		template <typename T> struct ConstFastMathVectorIterator
		{
		private:
			const MathVector<T>& parent;
			std::set<size_t>::const_iterator m_element_it;

		public:
			ConstFastMathVectorIterator(const MathVector<T>& _parent, size_t _position) 
			: parent(_parent)
			{
				if (_position == 0)
					m_element_it = parent.not_nulls.cbegin();
				else if (_position >= parent.size)
					m_element_it = parent.not_nulls.cend();
				else
				{
					m_element_it = parent.not_nulls.find(_position);
					if (m_element_it == parent.not_nulls.cend())
					{
						std::set<size_t>::const_iterator it = parent.not_nulls.cbegin();
						size_t min = parent.size;

						for (; it != parent.not_nulls.end(); ++it)
						{
							size_t distance = abs(*it - _position);
							if (distance <= min)
							{
								m_element_it = it;
								min = distance;
							}
						}
					}
				}
			}

			T getElem() const
			{ 
				return parent.getElement(*m_element_it); 
			}

			size_t index() const
			{
				return *m_element_it;
			}

			void operator++() { ++m_element_it; }
			void operator--() { --m_element_it; }

			bool operator==(const ConstFastMathVectorIterator& other) const { return m_element_it == other.m_element_it; }
			bool operator!=(const ConstFastMathVectorIterator& other) const { return !(*this == other); }
		};
	}
	}
}

#endif
