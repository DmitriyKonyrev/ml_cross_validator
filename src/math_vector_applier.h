#ifndef MATHVECTORAPPLIERS_H
#define MATHVECTORAPPLIERS_H

#include <functional>
#include <vector>

#include "common_vector.h"
#include "functional.h"
#include "math_vector.h"

namespace MathCore
{
    namespace AlgebraCore
    {
        template<typename T, typename R> class MathVectorOperatorApplier
        {
        public:
            std::vector<R> applySingleMapOperator(std::function<R (const T&)> map_operator, const MathVector<T>& vector)
            {
                return SingleMapper<T,R>(map_operator, vector.m_vectorData);
            }

            std::vector<R> applyPairMapOperator(std::function<R (const T&, const T&)> map_operator, const MathVector<T>& left, const MathVector<T>& right)
            {
                return PairMapper<T,R>(map_operator, left.m_vectorData, right.m_vectorData);
            }

            R applySingleReduceOperator(std::function<void (R&, const T&)> reduce_operator, const MathVector<T>& vector, R default_value = (R)0)
            {
                return SingleReducer<T,R>(reduce_operator, vector.m_vectorData, default_value);
            }

            R applyPairReduceOperator(std::function<void (R&, const T&, const T&)> reduce_operator, const MathVector<T>& left, const MathVector<T>& right, R default_value = (R)0)
            {
                return PairReducer<T,R>(reduce_operator, left.m_vectorData, right.m_vectorData);
            }

            std::vector<R> applySingleMapper(std::function<std::vector<R> (const CommonVector<T>*)> mapper, const MathVector<T>& vector)
            {
                return mapper(vector.m_vectorData);
            }

            std::vector<R> applyPairMapOperator(std::function<std::vector<R> (const CommonVector<T>*, const CommonVector<T>*)> mapper, const MathVector<T>& left, const MathVector<T>& right)
            {
                return mapper(left.m_vectorData, right.m_vectorData);
            }

            R applySingleReducer(std::function<R (const CommonVector<T>*)> reducer, const MathVector<T>& vector)
            {
                return reducer(vector.m_vectorData);
            }

            R applyPairReducer(std::function<R (const CommonVector<T>*, const CommonVector<T>*)> reducer, const MathVector<T>& left, const MathVector<T>& right)
            {
                return reducer(left.m_vectorData, right.m_vectorData);
            }

        public:
            MathVectorOperatorApplier() {};
        };
    }
}

#endif//MATHVECTORAPPLIERS_H
