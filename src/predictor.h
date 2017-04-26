#ifndef PREDICTOR_H
#define PREDICTOR_H

#include <vector>

#ifndef INSTANCE_H
#include "instance.h"
#endif

#ifndef METRIC_H
#include "metric.h"
#endif

namespace MachineLearning
{
	class Predictor
	{
		protected:

			size_t featuresCount;

		public:

			Predictor(size_t _featuresCount)
				: featuresCount(_featuresCount)
			{
			}


			virtual float predict(MathVector<float>& features);

			virtual void learn(std::vector<Instance>& learnSet, std::vector<std::pair<float, float>>& learning_curve);
			virtual std::vector<float> test(std::vector<Instance>& testSet, std::vector<Metrics::Metric>& metrics);

			size_t getFeaturesCount();
		protected:

			std::vector<float> rmse(std::vector<Instance>& instances);

	};

}

#endif
