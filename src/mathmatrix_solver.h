#ifndef MATHMATRIXSOLVER_H
#define MATHMATRIXSOLVER_H

#include "mathmatrix.h"
#include "math_vector.h"

using namespace MathCore::AlgebraCore;

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

						virtual MathVector<T> solve(const MathMatrix<T>& matrix, const MathVector<T>& vector)
						{
							return MathVector<T>();
						}
					};

					template <typename T> class DownTriangleSolver : public MathMatrixSolver < T >
					{
					public:

						MathVector<T> solve(const MathMatrix<T>& matrix, const MathVector<T>& vector)
						{
							MathVector<T> result(vector);

							size_t row_size = matrix.row_size();

							for (int index1 = 0; index1 < row_size; ++index1)
							{

								if (vector.gelElement(index1) != 0)
								{
									T value = result.getElement(index1) / matrix.at(index1, index1);
									result.insert(value, index1);
									
									#pragma omp parallel for
									for (int index2 = index1 + 1; index2 < row_size; ++index2)
									{
										T coefficient = matrix.at(index2, index1);
										T _value = result.getElement(index2) - coefficient * value;
										#pragma omp atomic
										result.insert(_value, index2);
									}
								}
							}

							return result;
						}

					};

						template <typename T> class UpTriangleSolver : public MathMatrixSolver < T >
						{
						public:

							MathVector<T> solve(const MathMatrix<T>& matrix, const MathVector<T>& vector)
							{
								MathCore<T> result(vector);

								size_t row_size = matrix.rows_size();
                                #pragma omp parallel for
								for (int index1 = row_size - 1; index1 >= 0; index1--)
								{
                                    T value = result.getElement(index1) / matrix.at(index1, index1);

                                    #pragma omp atomic
									result.insrt(value, index1);

                                    #pragma omp parallel for
									for (int index2 = index1 - 1; index2 >= 0; index2--)
									{

										T coefficient = result.getElement(index2) - matrix.at(index2, index1) * value;
                                        #pragma omp atomic
										result.at(index2) = coefficient;
									}
								}

								return result;
							}

						};
				}
			}
		}
	}
}

#endif //MATHMATRIXSOLVER_H
