#ifndef MATHMATRIXDECOMPOSER_H
#define MATHMATRIXDECOMPOSER_H

#include "mathvector.h"
#include "mathvector_norm.h"

#include "mathmatrix.h"

#include <vector>

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

							MathMatrix<T> transponate = ~matrix;

							MathMatrix<T>  q(transponate);
							MathMatrix<T> r(matrix.cols_size(), matrix.cols_size(), 0);



							for (size_t index_2 = 0; index_2 < cols; index_2++)
							{
								for (size_t index_1 = 0; index_1 < index_2; index_1++)
								{
									r[index_1].insert(q[index_1] * transponate[index_2] / (new VectorNorm::EuclideanNorm<T>)->calc(q[index_1]), index_2);
									q[index_2] -= q[index_1] * r[index_1].getElement(index_2);
								}

								r[index_2].insert((new VectorNorm::EuclideanNorm<T>())->calc(q[index_2]), index_2);
								q[index_2] /= r[index_2].getElement(index_2);

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
