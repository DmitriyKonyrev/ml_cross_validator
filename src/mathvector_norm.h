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
						typename MathVector<T>::fast_iterator firstEnd   = first.fast_end();

						typename MathVector<T>::fast_iterator secondBegin = second.fast_begin();
						typename MathVector<T>::fast_iterator secondEnd   = second.fast_end();

						T value = 0.;

						while (firstBegin != firstEnd || secondBegin != secondEnd)
						{
							if (firstBegin == firstEnd)
							{
								T newvalue = std::pow(secondBegin.getElem(), 2.);

								value += newvalue;

								++secondBegin;
							}
							else if (secondBegin == secondEnd)
							{
								T newvalue = std::pow(firstBegin.getElem(), 2.);

								value += newvalue;

								++firstBegin;
							}
							else if (firstBegin.index() == secondBegin.index())
							{
								T newvalue = std::pow(firstBegin.getElem() - secondBegin.getElem(), 2.);

								value += newvalue;

								++firstBegin;
								++secondBegin;
							}
							else if (firstBegin.index() > secondBegin.index())
							{
								T newvalue = std::pow(secondBegin.getElem(), 2.);

								value += newvalue;

								++secondBegin;
							}
							else if (firstBegin.index() < secondBegin.index())
							{
								T newvalue = std::pow(firstBegin.getElem(), 2.);

								value += newvalue;

								++firstBegin;
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
