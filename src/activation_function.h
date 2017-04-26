
#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include <cmath>

namespace MachineLearning
{
	struct ActivationFunction
	{
		public:

			virtual float calc(float value)
			{
				return 0;
			}

			virtual float dx(float value)
			{
				return 0;
			}
	};

	struct HeavisideActivationFunctionLogistic : public ActivationFunction
	{
	public:

		float calc(float value)
		{
			return value < 0 ? -1. : 1.;
		}

		float dx(float value)
		{
			return 1;
		}
	};

	struct HeavisideActivationFunction : public ActivationFunction
	{
		public:

			float calc(float value)
			{
				return value < 0 ? 0. : 1.;
			}

			float dx(float value)
			{
				return 1;
			}
	};

	struct SigmoidActivationFunction : public ActivationFunction
	{
		public:

			float calc(float value)
			{
				return pow(1 + exp(-value), -1);
			}

			float dx(float value)
			{
				float result = calc(value);

				return result * (1 - result);
			}
	};

	struct HyperbolicTanActivationFunction : public ActivationFunction
	{
		public:

			float sigma;

		public:

			float calc(float value)
			{
				return 2 * sigma * (2 * value) - 1;
			}

			float dx(float value)
			{
				return 4 * sigma;
			}
	};

	struct LogarithmicActivationFunction : public ActivationFunction
	{
		public:

			float calc(float value)
			{
				return log(value + sqrt(pow(value, 2) + 1));
			}

			float dx(float value)
			{
				return 1 / (value + sqrt(pow(value, 2) + 1)) * (1 + 0.5 * pow( pow(value, 2) + 1, -0.5) * 2 * value);
			}
	};

	struct GaussianActivationFunction : public ActivationFunction
	{
		public:

			float calc(float value)
			{
				return exp(-pow(value, 2) / 2);
			}

			float dx(float value)
			{
				return exp(-pow(value, 2) / 2) * (-value);
			}
	};

	struct LinearActivationFunction : public ActivationFunction
	{
		public:

			float calc(float value)
			{
				return value;
			}

			float dx(float value)
			{
				return 1;
			}
	};
}

#endif