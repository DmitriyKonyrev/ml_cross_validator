#include <iostream>
#include <random>
#include <ctime>
#include <vector>
#include <algorithm>
#include <math.h>

#include "ada_boost.h"

#include "predictor.h"
#include "instance.h"
#include "metric.h"

#include "mathvector.h"

using namespace MathCore::AlgebraCore::VectorCore;

namespace MachineLearning
{
	AdaBoost::AdaBoost( size_t _featuresCount
					  , PredictorPtr predictor_type
				      , size_t max_estimators
					  , Metrics::Metric quality_checker
				      , double max_quality
				      , bool bagging
					  , double bagging_factor)
	: Predictor(_featuresCount)
	, m_predictor_type(predictor_type)
	, m_max_estimators(max_estimators)
	, m_quality_checker(quality_checker)
	, m_max_quality(max_quality)
	, m_bagging(bagging)
	, m_bagging_factor(bagging_factor)
	{ }

	double AdaBoost::predict(MathVector<double>& features)
	{
		double prediction = 0;
#pragma omp parallel for reduction(+:prediction)
		for (size_t index = 0; index < m_estimators.size(); ++index)
		{
			prediction = prediction + m_weights[index] * m_estimators[index]->predict(features);
		}

		return prediction > 0.0 ? 1.0 : -1.0;
	}

	void AdaBoost::learn( std::vector<Instance>& learnSet
			            , std::vector<double>& objectsWeights
			            , std::vector<std::pair<double, double>>& learning_curve)
	{
		if (objectsWeights.empty())
			objectsWeights = std::vector<double>(learnSet.size(), 1.0);
		std::vector<Metrics::Metric> quality_func({m_quality_checker});
		double quality = test(learnSet, quality_func).front(); 
		size_t errors     = 0;
		size_t max_errors = 10;
		double norm_factor = 1.0 / (double)learnSet.size();
		std::vector<double> obj_weights = std::vector<double>(learnSet.size(), norm_factor);
		for (size_t index = 0; index < obj_weights.size(); ++index)
			obj_weights[index] *= objectsWeights[index];
		auto check_negative = [this]( PredictorPtr& predictor
				                    , std::vector<Instance>& learnSet
								    , std::vector<double>& objectsWeights)
	    -> double
		{
			double negative_predictions = 0.0;
#pragma omp parallel for reduction(+:negative_predictions)
			for (size_t obj_index = 0; obj_index < learnSet.size(); ++obj_index)
			{
				double error = (predictor->predict(learnSet[obj_index].getFeatures()) * learnSet[obj_index].getGoal() < 0.0);
				negative_predictions = negative_predictions + error * objectsWeights[obj_index];
			}

			return negative_predictions;
		};

		std::vector<double> objectsProbs(learnSet.size(), 1.0 / (double)learnSet.size());
		std::random_device rd;
		std::mt19937 gen(rd());
		std::discrete_distribution<> distribution(objectsProbs.begin(), objectsProbs.end());

		
		size_t estimator_index = 0;
		while (estimator_index < m_max_estimators)
		{
			std::cout << "train estimator #" << estimator_index << std::endl;
			PredictorPtr new_predictor {m_predictor_type->clone()};
			double negative_predictions = 0.0;
			if (!m_bagging)
			{
				new_predictor->learn(learnSet, obj_weights, learning_curve);
			}
			else
			{
				std::vector<Instance> subLearnSets;
				std::vector<double>   subsObjWeights;
				size_t bagging_size = learnSet.size() * m_bagging_factor;
				double summaries = 0.0;
				for (size_t obj_index = 0; obj_index < bagging_size; ++obj_index)
				{
					size_t instance_index = distribution(gen);
					subLearnSets.push_back(learnSet[instance_index]);
					subsObjWeights.push_back(obj_weights[instance_index]);
					summaries += obj_weights[instance_index];
				}
				
				new_predictor->learn(subLearnSets, subsObjWeights, learning_curve);
			}

			negative_predictions = check_negative(new_predictor, learnSet, obj_weights);
			double predictor_weight     = 0.5 * log((1.0 - negative_predictions + norm_factor) / (negative_predictions + norm_factor));
			std::cout << "\tnegative predictions: " << negative_predictions
			          << " predictor weight:" << predictor_weight << std::endl;
			std::cout << "\tstart normalizing weights" << std::endl;

			m_estimators.push_back(std::move(new_predictor));
			m_weights.push_back(predictor_weight);

			double summary = 0.0;
			for (size_t obj_index = 0; obj_index < learnSet.size(); ++obj_index)
			{
				double prediction = m_estimators.back()->predict(learnSet[obj_index].getFeatures());
				double assesment  = std::exp(prediction * learnSet[obj_index].getGoal() * (-predictor_weight));
				obj_weights[obj_index] *= assesment;
				summary = summary + obj_weights[obj_index];
			}

/*			double max_weight = *std::max_element(obj_weights.begin(), obj_weights.end());
			double min_weight = *std::min_element(obj_weights.begin(), obj_weights.end());
			double treshold   = max_weight - 0.001 * (max_weight - min_weight) / 2;
#pragma omp parallel for reduction(+:summary)
			for (double& weight: obj_weights)
			{
				if (weight >= treshold)
					weight = 0.0;
				summary = summary + weight;
			}*/

			for (double& weight: obj_weights)
				weight /= summary;

			double current_quality = test(learnSet, quality_func).front();
			std::cout << "\rlast quality: " << quality << " current quality: " << current_quality << std::endl;
			/*if (current_quality < quality)
			{
				m_estimators.pop_back();
				m_weights.pop_back();
				errors++;
				if (errors >= max_errors)
					break;
			}
			else
			{
				quality = current_quality;
				errors = 0;
				estimator_index++;
			}*/
			estimator_index++;
			if (current_quality >= m_max_quality)
				return;
		}

		std::cout << "learning finished"  << std::endl
			      << "\t   learn quality: " << quality                << std::endl
		          << "\testimators count: " << m_estimators.size()    << std::endl
				  << "\tmodel complexity: " << get_model_complexity() << std::endl;
	}
		
	size_t AdaBoost::get_model_complexity()
	{
		size_t model_complexity = 0;
		for (const PredictorPtr& predictor: m_estimators)
		{
			model_complexity += predictor->get_model_complexity();
		}

		return model_complexity;
	}

}
