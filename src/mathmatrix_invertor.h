#ifndef MATHMATRIXINVERTOR_H
#define MATHMATRIXINVERTOR_H

#include <iostream>
#include <math.h>

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

                            std::cout << "start decomposing" << std::endl;
							std::vector<MathMatrix<T>> decompose = (new MatrixDecomposer::QRDecomposerGS<T>())->decompose(matrix);

							q = decompose.at(0);
							r = decompose.at(1);

							size_t row_size = matrix.rows_size();
                            std::vector<MathVector<T>> inverse_r_raw(row_size);

                            size_t total_count = 0;
                            size_t part = pow(10, 3);
                            std::cout << "start inverting R" << std::endl;

                            #pragma omp parallel for
							for (size_t index = 0; index < row_size; index++)
							{
								MathVector<T> step = (new MatrixSolver::UpTriangleSolver<T>())->solve(r, e[index]);
								inverse_r_raw.at(index) = step;

                                #pragma omp atomic
                                total_count++;
                                #pragma omp atomic
                                if ((total_count % part) == 0)
                                    std::cout << "total count: " << total_count << std::endl;
							}

							MathMatrix<T> inverse_r(inverse_r_raw);

                            std::cout << "final inverting" << std::endl;
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
