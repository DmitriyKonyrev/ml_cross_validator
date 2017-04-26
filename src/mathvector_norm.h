#ifndef MATHVECTORNORM_H
#define MATHVECTORNORM_H

#include "mathvector.h"

namespace MathCore
{
	namespace AlgebraCore
	{
		namespace VectorCore
		{
			namespace VectorNorm
			{
				template <typename T> class  MathVectorNorm
				{
				public:

					MathVectorNorm()
					{

					}

					virtual T calc(MathVector<T>& vector)
					{
						return 0;
					}

					virtual T calc(MathVector<T>& first, MathVector<T>& second)
					{
						return 0;
					}
				};


				template <typename T> class EuclideanNorm : public MathVectorNorm < T >
				{
				public:

					EuclideanNorm()
				    : MathVectorNorm<T>()
					{

					}

					T calc(MathVector<T>& vector)
					{
						T norm = vector * vector;

						return sqrt(norm);
					}

					T calc(MathVector<T>& first, MathVector<T>& second)
					{
						typename MathVector<T>::fast_iterator firstBegin = first.fast_begin();
						typename MathVector<T>::fast_iterator secondBegin = second.fast_begin();


						T value = 0.;

						while (firstBegin != first.fast_end() || secondBegin != second.fast_end())
						{
							if (firstBegin == first.fast_end())
							{
								T newvalue = std::pow(secondBegin->second, 2.);

								value += newvalue;

								secondBegin++;
							}
							else if (secondBegin == second.fast_end())
							{
								T newvalue = std::pow(firstBegin->second, 2.);

								value += newvalue;

								firstBegin++;
							}
							else if (firstBegin->first == secondBegin->first)
							{
								T newvalue = std::pow(firstBegin->second - secondBegin->second, 2.);

								value += newvalue;


								firstBegin++;
								secondBegin++;
							}
							else if (firstBegin->first > secondBegin->first)
							{
								T newvalue = std::pow(secondBegin->second, 2.);

								value += newvalue;

								secondBegin++;
							}
							else if (firstBegin->first < secondBegin->first)
							{
								T newvalue = std::pow(firstBegin->second, 2.);

								value += newvalue;

								firstBegin++;
							}

						}


						return std::pow(value, 0.5);
					}
				};
			}
		}
	}
}

#endif//MATHVECTORNORM_H
