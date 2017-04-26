
#ifndef LOSS_FUNCTION_APPROXIMATION_H
#define LOSS_FUNCTION_APPROXIMATION_H

#include <cmath>

namespace MachineLearning
{
	struct LossFunctionApproximation
	{
		public:

			virtual float calc(float margin)
			{
				return 0;
			}

			virtual float dx(float margin)
			{
				return 0;
			}
	};

	struct QuadraticLossFunction : public LossFunctionApproximation
	{
		public:

			float calc(float margin)
			{
				return pow(margin, 2);
			}

			float dx(float margin)
			{
				return margin;
			}
	};

	struct PiecewiseLinearLossFunction : public LossFunctionApproximation
	{
		public:

			float calc(float margin)
			{
				return 1 - margin;
			}

			float dx(float margin)
			{
				return - margin;
			}
	};

	struct SigmoidLossFunction : public LossFunctionApproximation
	{
		public:

			float calc(float margin)
			{
				return 2 * pow((1 + std::exp(margin)), -1);
			}

			float dx(float margin)
			{
				return -2 * pow((1 + std::exp(margin)), -2);
			}
	};

	struct LogisticLossFunction : public LossFunctionApproximation
	{
		public:

			float calc(float margin)
			{
				return 2 * log2(1 + std::exp(-margin));
			}

			float dx(float margin)
			{
				return -std::exp(-margin) / ((1 + std::exp(-margin)) + log(2));
			}
	};

	struct ExpLossFunction : public LossFunctionApproximation
	{
		public:

			float calc(float margin)
			{
				return std::exp(-margin);
			}

			float dx(float margin)
			{
				return -std::exp(-margin);
			}
	};

}

#endif