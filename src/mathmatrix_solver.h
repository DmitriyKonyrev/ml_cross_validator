#ifndef MATHMATRIXSOLVER_H
#define MATHMATRIXSOLVER_H

#include "mathmatrix.h"


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
				namespace MatrixSolver
				{
					template <typename T> class MathMatrixSolver
					{
					public:

						virtual MathVector<T> solve(MathMatrix<T>& matrix, MathVector<T>& vector)
						{
							return MathVector<T>();
						}
					};

					template <typename T> class DownTriangleSolver : public MathMatrixSolver < T >
					{
					public:

						MathVector<T> solve(MathMatrix<T>& matrix, MathVector<T>& vector)
						{
							std::vector<T> result(vector.to_std_vector());

							size_t row_size = matrix.row_size();

							for (int index1 = 0; index1 < row_size; ++index1)
							{

								if (result.at(index1) != 0)
								{
									result.at(index1) /= matrix.at(index1, index1);

									T value = result.at(index1);

#pragma omp parallel for private(value);
									for (int index2 = index1 + 1; index2 < row_size; ++index2)
									{
										T coefficient = matrix.at(index2, index1);

										result.at(index2) -= coefficient * value;
									}

								}
							}

							return MathVector<T>(result);
						}

					};

						template <typename T> class UpTriangleSolver : public MathMatrixSolver < T >
						{
						public:

							MathVector<T> solve(MathMatrix<T>& matrix, MathVector<T>& vector)
							{
								std::vector<T> result(vector.to_std_vector());

								size_t row_size = matrix.rows_size();
                                #pragma omp parallel for
								for (int index1 = row_size - 1; index1 >= 0; index1--)
								{
                                    T value = result.at(index1) / matrix.at(index1, index1);

                                    #pragma omp atomic
									result.at(index1) = value;

                                    #pragma omp parallel for
									for (int index2 = index1 - 1; index2 >= 0; index2--)
									{

										T coefficient = result.at(index2) - matrix.at(index2, index1) * value;
                                        #pragma omp atomic
										result.at(index2) = coefficient;
									}
								}

								return MathVector<T>(result);
							}

						};
				}
			}
		}
	}
}

#endif //MATHMATRIXSOLVER_H
