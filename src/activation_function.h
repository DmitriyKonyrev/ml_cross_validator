
#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include <cmath>

namespace MachineLearning
{
	struct ActivationFunction
	{
		public:

			virtual double calc(double value)
			{
				return 0;
			}

			virtual double dx(double value)
			{
				return 0;
			}
	};

	struct HeavisideActivationFunctionLogistic : public ActivationFunction
	{
	public:

		double calc(double value)
		{
			return value < 0 ? -1. : 1.;
		}

		double dx(double value)
		{
			return 1;
		}
	};

	struct HeavisideActivationFunction : public ActivationFunction
	{
		public:

			double calc(double value)
			{
				return value < 0 ? 0. : 1.;
			}

			double dx(double value)
			{
				return 1;
			}
	};

	struct SigmoidActivationFunction : public ActivationFunction
	{
		public:

			double calc(double value)
			{
				return pow(1 + exp(-value), -1);
			}

			double dx(double value)
			{
				double result = calc(value);

				return result * (1 - result);
			}
	};

	struct HyperbolicTanActivationFunction : public ActivationFunction
	{
		public:

			double sigma;

		public:

			double calc(double value)
			{
				return 2 * sigma * (2 * value) - 1;
			}

			double dx(double value)
			{
				return 4 * sigma;
			}
	};

	struct LogarithmicActivationFunction : public ActivationFunction
	{
		public:

			double calc(double value)
			{
				return log(value + sqrt(pow(value, 2) + 1));
			}

			double dx(double value)
			{
				return 1 / (value + sqrt(pow(value, 2) + 1)) * (1 + 0.5 * pow( pow(value, 2) + 1, -0.5) * 2 * value);
			}
	};

	struct GaussianActivationFunction : public ActivationFunction
	{
		public:

			double calc(double value)
			{
				return exp(-pow(value, 2) / 2);
			}

			double dx(double value)
			{
				return exp(-pow(value, 2) / 2) * (-value);
			}
	};

	struct LinearActivationFunction : public ActivationFunction
	{
		public:

			double calc(double value)
			{
				return value;
			}

			double dx(double value)
			{
				return 1;
			}
	};
}

#endif