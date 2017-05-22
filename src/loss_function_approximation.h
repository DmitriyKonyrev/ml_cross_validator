
#ifndef LOSS_FUNCTION_APPROXIMATION_H
#define LOSS_FUNCTION_APPROXIMATION_H

#include <cmath>

namespace MachineLearning
{
	struct LossFunctionApproximation
	{
		public:

			virtual double calc(double margin)
			{
				return 0;
			}

			virtual double dx(double margin)
			{
				return 0;
			}
	};

	struct QuadraticLossFunction : public LossFunctionApproximation
	{
		public:

			double calc(double margin)
			{
				return pow(margin, 2);
			}

			double dx(double margin)
			{
				return margin;
			}
	};

	struct PiecewiseLinearLossFunction : public LossFunctionApproximation
	{
		public:

			double calc(double margin)
			{
				return 1 - margin;
			}

			double dx(double margin)
			{
				return - margin;
			}
	};

	struct SigmoidLossFunction : public LossFunctionApproximation
	{
		public:

			double calc(double margin)
			{
				return 2 * pow((1 + std::exp(margin)), -1);
			}

			double dx(double margin)
			{
				return -2 * pow((1 + std::exp(margin)), -2);
			}
	};

	struct LogisticLossFunction : public LossFunctionApproximation
	{
		public:

			double calc(double margin)
			{
				return 2 * log2(1 + std::exp(-margin));
			}

			double dx(double margin)
			{
				return -std::exp(-margin) / ((1 + std::exp(-margin)) + log(2));
			}
	};

	struct ExpLossFunction : public LossFunctionApproximation
	{
		public:

			double calc(double margin)
			{
				return std::exp(-margin);
			}

			double dx(double margin)
			{
				return -std::exp(-margin);
			}
	};

}

#endif