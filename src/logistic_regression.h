#ifndef LOGISTIC_REGRESSION_CLASSIFIER_H
#define LOGISTIC_REGRESSION_CLASSIFIER_H


#include <vector>


#include "predictor.h"
#include "instance.h"
#include "metric.h"
#include "loss_function_approximation.h"
#include "activation_function.h"

#include "math_vector.h"

using namespace MathCore::AlgebraCore;

namespace MachineLearning
{
	class LogisticRegression : public Predictor
	{
		protected:

			MathVector<float> weights;

			float threshold;

			HeavisideActivationFunctionLogistic* activate = new HeavisideActivationFunctionLogistic();

			SigmoidActivationFunction* learningActivate = new SigmoidActivationFunction();

			LogisticLossFunction* approximation = new LogisticLossFunction();

			size_t minimalIterations;
			size_t maximalIterations;

		public:

			LogisticRegression(size_t _featuresCount, size_t _minimalIterations, size_t _maximalIterations)
			: Predictor(_featuresCount)
            , minimalIterations(_minimalIterations)
            , maximalIterations(_maximalIterations)
			{
			}

			float predict(MathVector<float>& features);
			void learn(std::vector<Instance>& learnSet, std::vector<std::pair<float, float>>& learning_curve);
			float quality(std::vector<Instance>& testSet);
			void setIterationInterval(size_t _minimalIterations, size_t _maximalIterations);

		private:
			float scalarProduct(MathVector<float>& features);
			float predictRaw(float _scalar);
			MathVector<float>& weightsInit(size_t size);
			void weightsJog();
	};
}

#endif
