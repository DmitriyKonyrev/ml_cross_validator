#ifndef MATHMATRIXINVERTOR_H
#define MATHMATRIXINVERTOR_H

#include "mathmatrix.h"
#include "mathmatrix_decomposer.h"
#include "mathmatrix_solver.h"

#include "mathvector.h"

using namespace MathCore::AlgebraCore::VectorCore;

namespace MathCore
{
	namespace AlgebraCore
	{
		namespace MatrixCore
		{
			template<typename T> class MathMatrix;

			namespace MatrixAlgorithm
			{
				namespace MatrixInvertor
				{

					template <typename T> class MathMatrixInvertor
					{
					public:

						virtual MathMatrix<T>& invert(MathMatrix<T>& matrix)
						{
							return matrix;
						}
					};



					template <typename T> class QRInvertor : public MathMatrixInvertor < T >
					{
					public:

						MathMatrix<T>& invert(MathMatrix<T>& matrix)
						{
							MathMatrix<T> e = MathMatrix<T>::createIdentityMatrix(matrix.cols_size(), 1);

							MathMatrix<T> q;
							MathMatrix<T> r;

							std::vector<MathMatrix<T>> decompose = (new MatrixDecomposer::QRDecomposerGS<T>())->decompose(matrix);

							q = decompose.at(0);
							r = decompose.at(1);

							MathMatrix<T> inverse_r(0, matrix.cols_size(), 0);

							size_t row_size = matrix.rows_size();

							for (size_t index = 0; index < row_size; index++)
							{
								MathVector<T> step = (new MatrixSolver::UpTriangleSolver<T>())->solve(r, e[index]);
								inverse_r.push_back(step);
							}

							MathMatrix<T>* inverse_t = &(~(q * inverse_r));

							return *inverse_t;
						}
					};
				}
			}
		}
	}
}

#endif// MATHMATRIXINVERTOR_H
