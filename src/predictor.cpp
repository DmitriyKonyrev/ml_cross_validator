#include <vector>
#include <math.h>

#ifndef INSTANCE_H
#include "instance.h"
#endif

#ifndef PREDICTOR_H
#include "predictor.h"
#endif

using namespace MachineLearning;

std::vector<float> Predictor::rmse(std::vector<Instance>& instances)
{
	float sumSquaredError = 0.;

	float true_positive = 0.;
	float false_positive = 0.;
	float true_negative = 0.;
	float false_negative = 0.;

	for (size_t index = 0; index < instances.size(); index++)
	{

		float prediction = this->predict(instances.at(index).getFeatures());


		if (prediction == instances.at(index).getGoal())
		{
			if (prediction == 1)
			{
				true_positive++;
			}
			else
			{
				true_negative++;
			}
		}
		else
		{
			if (prediction == 1)
			{
				false_positive++;
			}
			else
			{
				false_negative++;
			}
		}
	}

	sumSquaredError /= instances.size();

	std::vector<float>* characteristics = new std::vector<float>();
	characteristics->push_back(std::pow(sumSquaredError, 0.5));
	characteristics->push_back(true_positive);
	characteristics->push_back(false_positive);
	characteristics->push_back(true_negative);
	characteristics->push_back(false_negative);

	return *characteristics;
}

float Predictor::predict(MathVector<float>& features)
{
	return 0;
}

void Predictor::learn(std::vector<Instance>& learnSet, std::vector<std::pair<float, float>>& learning_curve)
{
	return;
}

std::vector<float> Predictor::test(std::vector<Instance>& learnSet, std::vector<Metrics::Metric>& metrics)
{
	std::vector<float>* results = new std::vector<float>();
	float sumSquaredError = 0.;

	float true_positive = 0.;
	float false_positive = 0.;
	float true_negative = 0.;
	float false_negative = 0.;

	for (size_t index = 0; index < learnSet.size(); index++)
	{

		float prediction = this->predict(learnSet.at(index).getFeatures());
		sumSquaredError += std::pow(prediction - learnSet.at(index).getGoal(), 2);

		if (prediction == learnSet.at(index).getGoal())
		{
			if (prediction == 1)
			{
				true_positive++;
			}
			else
			{
				true_negative++;
			}
		}
		else
		{
			if (prediction == 1)
			{
				false_positive++;
			}
			else
			{
				false_negative++;
			}
		}
	}

    sumSquaredError /= learnSet.size();
	results->clear();

	for (size_t index = 0; index < metrics.size(); index++)
	{
		float result = metrics.at(index)(true_positive, false_positive, true_negative, false_negative);

		results->push_back(result);
	}

    results->push_back(sumSquaredError);

	return *results;
}



size_t Predictor::getFeaturesCount()
{
	return this->featuresCount;
}
