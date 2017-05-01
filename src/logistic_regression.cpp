#include <iostream>
#include <random>
#include <ctime>
#include <vector>
#include <algorithm>
#include <math.h>

#ifndef PREDICTOR_H
#include "predictor.h"
#endif

#ifndef INSTANCE_H
#include "instance.h"
#endif

#ifndef METRIC_H
#include "metric.h"
#endif

#ifndef LOSS_FUNCTION_APPROXIMATION_H
#include "loss_function_approximation.h"
#endif

#ifndef ACTIVATION_FUNCTION_H
#include "activation_function.h"
#endif

#ifndef LOGISTIC_REGRESSION_CLASSIFIER_H
#include "logistic_regression.h"
#endif

#include "math_vector.h"
#include "math_vector_norm.h"

using namespace MachineLearning;
using namespace MathCore::AlgebraCore;

float LogisticRegression::scalarProduct(MathVector<float>& features)
{

	float product = features * this->weights;

	product -= this->threshold;

	return product;

}

float LogisticRegression::predictRaw(float _scalar)
{
	return this->activate->calc(_scalar);
}

float LogisticRegression::predict(MathVector<float>& features)
{
	float _product = this->scalarProduct(features);

	return this->activate->calc(_product);
}

void LogisticRegression::setIterationInterval(size_t _minimalIterations, size_t _maximalIterations)
{
	this->minimalIterations = _minimalIterations;
	this->maximalIterations = _maximalIterations;

	return;
}

float LogisticRegression::quality(std::vector<Instance>& testSet)
{
	float _summary = 0;

	for (size_t index = 0; index < testSet.size(); index++)
	{
		float _margin = this->scalarProduct(testSet.at(index).getFeatures()) * testSet.at(index).getGoal();

		_summary += this->approximation->calc(_margin);
	}

	return _summary;
}

MathVector<float>& LogisticRegression::weightsInit(size_t size)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> distribution(-1 / (float)size, 1 / (float)size);

	std::vector<float> weights;

	for (size_t index = 0; index < size; index++)
	{
		weights.push_back(distribution(gen));
	}

	this->threshold = distribution(gen);

	this->weights = MathVector<float>(weights);

	return this->weights;
}


void LogisticRegression::learn(std::vector<Instance>& learnSet, std::vector<std::pair<float, float>>& learning_curve)
{
	float learning_rate = 0.01;

	size_t length = learnSet.size();

	size_t features_count = learnSet.at(0).getFeatures().get_dimension();

	this->weights = MathVector<float>(features_count, 0);

	this->threshold = 0;

	float lambda = 1 / (float)learnSet.size();

	std::srand(unsigned(std::time(NULL)));

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<size_t> distribution(0, learnSet.size() - 1);

	float q_assessment = this->quality(learnSet);

	float q_assessment_last = q_assessment;

	float weight_difference = 0;

	size_t iterations = 0;

    size_t part = pow(10, 3);

	do
	{
		size_t instance_index = distribution(gen);
		float _scalar = this->scalarProduct(learnSet.at(instance_index).getFeatures());
		float _prediction = this->predictRaw(_scalar);
		float _real_value = learnSet.at(instance_index).getGoal();
		float _margin = _scalar * _real_value;
		float _error = this->approximation->calc(_margin);

		float _activation_learn = this->learningActivate->calc(-_margin);
		float factor = learning_rate * _activation_learn * _real_value;

		weight_difference = 0.0;
        weight_difference = this->weights.update(factor, learnSet.at(instance_index).getFeatures());
		float new_weight_value = threshold - factor;
		weight_difference += pow(abs(new_weight_value - threshold),2.);

		this->threshold = new_weight_value;

		weight_difference /= (float)featuresCount;
		weight_difference = pow(weight_difference, 0.5);

		q_assessment_last = q_assessment;

		q_assessment = (1 - lambda) * q_assessment_last + lambda * _error;

		iterations++;
		if (iterations % length == 0)
		{
			std::cout << "iterations  : " << iterations <<  std::endl
			  << "\tq_assessment_diff : " << abs(q_assessment - q_assessment_last) << std::endl
			  << "\tweight_difference : " << weight_difference << std::endl
              << "\tmodel complexity  : " << this->weights.getSizeOfNotNullElements() << std::endl
			  << "\ttreshold          : " << threshold << std::endl;

		}
        if (iterations % part == 0)
            learning_curve.push_back(std::make_pair(_scalar, _real_value));
	}
   	while ((abs(q_assessment - q_assessment_last) >= pow(10,   -5) ||
			weight_difference >= pow(10, -5) ||
			iterations <= minimalIterations * length) &&
			iterations <= maximalIterations * length);
    std::cout << "Total characteristics:" << std::endl;
	std::cout << "\ttotal iterations  : " << iterations <<  std::endl
			  << "\tq_assessment_diff : " << abs(q_assessment - q_assessment_last) << std::endl
			  << "\tweight_difference : " << weight_difference << std::endl
              << "\tmodel complexity  : " << this->weights.getSizeOfNotNullElements() << std::endl
			  << "\ttreshold          : " << threshold << std::endl;
	return;
}

void LogisticRegression::weightsJog()
{
	std::random_device rd;
	std::mt19937 gen(rd());

	size_t size = this->weights.get_dimension();

	std::uniform_real_distribution<float> distribution(-1, 1);

	for (size_t index = 0; index < size; index++)
	{
		this->weights.insert(this->weights.getElement(index) + distribution(gen), index);
	}

	this->threshold += distribution(gen);

	return;
}
