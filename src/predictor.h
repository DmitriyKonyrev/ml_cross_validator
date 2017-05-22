#ifndef PREDICTOR_H
#define PREDICTOR_H

#include <vector>
#include <memory>

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


			virtual double predict(MathVector<double>& features);

			virtual void learn( std::vector<Instance>& learnSet
					          , std::vector<double>& objectsWeights
					          , std::vector<std::pair<double, double>>& learning_curve);
			virtual std::vector<double> test(std::vector<Instance>& testSet, std::vector<Metrics::Metric>& metrics);

			size_t getFeaturesCount();
			virtual size_t get_model_complexity() = 0;

			virtual Predictor* clone() const = 0;
		protected:

			std::vector<double> rmse(std::vector<Instance>& instances);

	};

	typedef std::shared_ptr<Predictor> PredictorPtr;

}

#endif
