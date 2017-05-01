#ifndef COMMONVECTORINITIALIZER_H
#define COMMONVECTORINITIALIZER_H

#include <algorithm>
#include <map>
#include <memory>
#include <vector>

#include "common_vector.h"
#include "dense_vector.h"
#include "sparse_vector.h"

namespace MathCore
{
    namespace AlgebraCore
    {
        namespace CommonVectorInitializer
        {

            template <typename T> std::shared_ptr<CommonVector<T>> InitializeCommonVector(size_t size, T default_value=(T)0)
            {
                if (default_value == (T)0)
                {
                    return std::shared_ptr<CommonVector<T>>(new SparseVector<T>(size, default_value));
                }
                else
                {
                    return std::shared_ptr<CommonVector<T>>(new DenseVector<T>(size, default_value));
                }
            }

            template <typename T> std::shared_ptr<CommonVector<T>> InitializeCommonVector(const std::vector<T>& values)
            {
                size_t notNullCount = 0;
                for (size_t index = 0; index < values.size(); ++index)
                {
                    if (values.at(index) != 0)
                        notNullCount++;
                }

                float loadFactor = (float)notNullCount / (float)values.size();
                if (loadFactor < 0.4)
                {
                    return std::shared_ptr<CommonVector<T>>(new SparseVector<T>(values));
                }
                else
                {
                    return std::shared_ptr<CommonVector<T>>(new DenseVector<T>(values));
                }
            }

            template <typename T> std::shared_ptr<CommonVector<T>> InitializeCommonVector(const std::map<size_t, T>& mapped_values,
                                                                            size_t fullSize)
            {
                float loadFactor = (float)mapped_values.size() / (float)fullSize;
                if (loadFactor < 0.4)
                {
                    return std::shared_ptr<CommonVector<T>>(new SparseVector<T>(mapped_values, fullSize));
                }
                else
                {
                    return std::shared_ptr<CommonVector<T>>(new DenseVector<T>(mapped_values, fullSize));
                }
            }
        }
    }
}

#endif//COMMONVECTORINITIALIZER_H
