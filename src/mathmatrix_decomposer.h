#ifndef MATHMATRIXDECOMPOSER_H
#define MATHMATRIXDECOMPOSER_H

#include "mathvector.h"
#include "mathvector_norm.h"

#include "mathmatrix.h"

#include <iostream>
#include <vector>
#include <math.h>

using namespace MathCore::AlgebraCore::VectorCore;
using namespace MathCore::AlgebraCore::VectorCore::VectorNorm;

namespace MathCore
{
	namespace AlgebraCore
	{
		namespace MatrixCore
		{
			template<typename T> class MathMatrix;

			namespace MatrixAlgorithm
			{
				namespace MatrixDecomposer
				{
					template <typename T> class MathMatrixDecomposer
					{

					public:

						virtual std::vector<MathMatrix<T>> decompose(MathMatrix<T> matrix)
						{
							return std::vector<MathMatrix<T>>();
						}

					};

					template <typename T> class QRDecomposerGS : public MathMatrixDecomposer < T >
					{

					public:

						std::vector<MathMatrix<T>> decompose(MathMatrix<T> matrix)
						{
							size_t rows = matrix.rows_size();
							size_t cols = matrix.cols_size();

                            VectorNorm::EuclideanNorm<T> euclidean;

							MathMatrix<T> transponate = ~matrix;

							MathMatrix<T>  q(transponate);
							MathMatrix<T> r(matrix.cols_size(), matrix.cols_size(), 0);
                            size_t total_count = 0;
                            size_t part = pow(10, 2);

							for (size_t index_2 = 0; index_2 < cols; index_2++)
							{
								T regular = q[index_2].getElement(index_2) + pow(10.0, -6);
								q[index_2].insert(regular, index_2);
								T r_value_ = euclidean.calc(q[index_2]);
								r[index_2].insert(r_value_, index_2);
								q[index_2] /= r[index_2].getElement(index_2);
								#pragma omp parallel for
								for (size_t index_1 = index_2 + 1; index_1 < cols; index_1++)
								{
                                    T r_value = q[index_2] * q[index_1];
									q[index_1] -= q[index_2] * r_value;
									r[index_2].insert(r_value, index_1);
								}

								total_count++;
								if ((total_count % part) == 0)
									std::cout << "processed: " << total_count << " of " << cols << std::endl;
							}

							std::vector<MathMatrix<T>>* result = new std::vector<MathMatrix<T>>();

							result->push_back(~q);
							result->push_back(r);


							return *result;
						}

					};
				}
			}

		}
	}
}

#endif//MATHMATRIXDECOMPOSER_H
