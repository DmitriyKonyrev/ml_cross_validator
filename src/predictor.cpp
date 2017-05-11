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
				true_positive = true_positive + 1;
			}
			else
			{
				true_negative = true_negative + 1;
			}
		}
		else
		{
			if (prediction == 1)
			{
				false_positive = false_positive + 1;
			}
			else
			{
				false_negative = false_negative + 1;
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
	
	size_t total_count = 0;
	size_t part = 1e2;

#pragma omp parallel for reduction (+:true_positive,false_positive,true_negative,false_negative,sumSquaredError)
	for (size_t index = 0; index < learnSet.size(); index++)
	{

		float prediction = this->predict(learnSet.at(index).getFeatures());
		float sse = std::pow(prediction - learnSet.at(index).getGoal(), 2);
		sumSquaredError = sumSquaredError + sse;

		if (prediction == learnSet.at(index).getGoal())
		{
			if (prediction == 1)
			{
				true_positive = true_positive + 1;
			}
			else
			{
				true_negative = true_negative + 1;
			}
		}
		else
		{
			if (prediction == 1)
			{
				false_positive = false_positive + 1;
			}
			else
			{
				false_negative = false_negative + 1;
			}
		}

#pragma omp atomic
		total_count++;
#pragma omp atomic
		if (total_count % part == 0)
			std::cout << "processed " << total_count << " objects" << std::endl;
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
