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

#include "mathvector_norm.h"

using namespace MachineLearning;
using namespace MathCore::AlgebraCore::VectorCore::VectorNorm;

float LogisticRegression::scalarProduct(MathVector<float>& features)
{

	float product = features * this->weights;

	product -= this->threshold;

	return product;

}

float LogisticRegression::predictRaw(float _scalar)
{
	//return this->activate->calc(_scalar);
	return this->learningActivate->calc(_scalar) * 2. - 1.;
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
		this->weights.push_back(distribution(gen));
	}

	this->threshold = distribution(gen);

	this->weights.setValues(weights);

	return this->weights;
}


void LogisticRegression::learn(std::vector<Instance>& learnSet, std::vector<std::pair<float, float>>& learning_curve)
{
	float learning_rate = default_learning_rate;
	size_t length = learnSet.size();

	size_t features_count = learnSet.at(0).getFeatures().getSize();
	
	this->weight_init(this->weights, this->threshold, learnSet);

	float lambda = 0.0;
	if (learnSet.size() < 5 * 1e3)
		lambda = 1 / (float)learnSet.size();
	else
		lambda = pow(10.0, log10((float)learnSet.size()) - 2.0) / (float)learnSet.size();
	std::cout << "lambda: " << lambda << std::endl;
	std::srand(unsigned(std::time(NULL)));

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<size_t> distribution(0, learnSet.size() - 1);

	float q_assessment = this->quality(learnSet);
	float max_q_assessment = q_assessment;
	std::cout << q_assessment << std::endl;

	float q_assessment_last = q_assessment;
	float weight_difference = 0;
	size_t iterations = 0;
    size_t part = pow(10, 3);
	size_t window = pow(10, 2);

	float accumulated_logloss = 0.0;
	float accumulated_rmse    = 0.0;

	bool last_q_assesment_inf = false;

	EuclideanNorm<float> euclidean;
	std::vector<Metrics::Metric> jog_metric({Metrics::F1ScoreMetric});
	float f1_quality           = test(learnSet, jog_metric).front();

	float precision = 1e-4;
	if (do_auto_precision)
	{
		size_t positive_count = 0;
		for (Instance& object: learnSet)
		{
			if (object.getGoal() == 1.0)
				positive_count++;
		}

		float volume = (float)positive_count / (float)learnSet.size();
		volume = std::min(volume, (float)1.0 - volume);
		if (volume < 0.3)
		{
			precision = 1e-3;
		}
		else if (volume < 0.15)
		{
			precision = 1e-2;
		}

		std::cout << "enabled auto-precision:" << precision << std::endl;
	}

    std::cout << "Euclidean norm:" << euclidean.calc(weights) << " Treshold:" << threshold << std::endl;

	do
	{
		size_t instance_index = distribution(gen);
		float _scalar = this->scalarProduct(learnSet.at(instance_index).getFeatures());

		if (learning_rate_type == LearningRateTypes::DIV)
		{
			learning_rate /= (iterations + 1);
		}
		else if (learning_rate_type == LearningRateTypes::EUCLIDEAN)
		{
			learning_rate = euclidean.calc(learnSet.at(instance_index).getFeatures()) * default_learning_rate;
		}

		float _prediction = this->predictRaw(_scalar);
		float _real_value = learnSet.at(instance_index).getGoal();
		accumulated_logloss += Metrics::Logloss(_real_value, _prediction);
		accumulated_rmse    += Metrics::RMSE(_real_value, _prediction);

		float _margin = _scalar * _real_value;
		float _error = this->approximation->calc(_margin);

		float _activation_learn = this->learningActivate->calc(-_margin);
		float regularize_factor = 1.0 - learning_rate * tau;

		float factor = learning_rate * _activation_learn * _real_value;

		weight_difference = 0.0;
        weight_difference = this->weights.update(regularize_factor, factor, learnSet.at(instance_index).getFeatures());
		float new_weight_value = regularize_factor * threshold - factor;
		weight_difference += pow(abs(new_weight_value - threshold),2.);

		this->threshold = new_weight_value;

		weight_difference /= (float)featuresCount;
		weight_difference = pow(weight_difference, 0.5);
		q_assessment = isinf(q_assessment) ? max_q_assessment : q_assessment;
		last_q_assesment_inf = isinf(q_assessment);
		q_assessment_last = q_assessment;
		q_assessment = (1 - lambda) * q_assessment_last + lambda * _error;

		iterations++;


		if (iterations % length == 0)
		{
			if (learning_rate_type == LearningRateTypes::DIV)
			{
				learning_rate = default_learning_rate;
			}

			if (do_jogging)
			{
				float cur_f1_quality = test(learnSet, jog_metric).front();
				std::cout << " last F1: " << f1_quality << " current F1:" << cur_f1_quality << std::endl;
				if (cur_f1_quality < f1_quality)
				{
					weightsJog();
				}
				else
				{
					f1_quality = cur_f1_quality;
				}
			}

			std::cout << "iterations: " << iterations << " weight_difference: " << weight_difference << " q_assessment_diff: " << abs(q_assessment - q_assessment_last) << std::endl;
		}
        if (iterations % window == 0)
		{
            learning_curve.push_back(std::make_pair(accumulated_logloss / iterations, accumulated_rmse / iterations));
		}
	}
   	while (((abs(q_assessment - q_assessment_last) >= precision ||
			weight_difference >= precision) ||
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

	size_t size = this->weights.getSize();
	EuclideanNorm<float> euclidean;
	float norm = euclidean.calc(this->weights);
	std::uniform_real_distribution<float> distribution(-1 / (float)size * norm, 1 /(float)size * norm);
	std::vector<float> jog_values(size, 0.0);
	for (size_t index = 0; index < size; index++)
	{
		jog_values[index] = distribution(gen);
		//this->weights.insert(this->weights.getElement(index) + distribution(gen), index);
	}
	this->weights += MathVector<float>(jog_values);
	this->threshold += distribution(gen);

	return;
}

size_t LogisticRegression::get_model_complexity()
{
	return this->weights.getSizeOfNotNullElements();
}
