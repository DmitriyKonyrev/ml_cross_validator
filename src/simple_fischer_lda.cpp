#include <algorithm>
#include <iostream>
#include <vector>
#include <math.h>

#include "simple_fischer_lda.h"


#ifndef PREDICTOR_H
#include "predictor.h"
#endif

#ifndef INSTANCE_H
#include "instance.h"
#endif

#include "mathmatrix.h"
#include "math_vector.h"
#include "math_vector_norm.h"

using namespace MathCore::AlgebraCore;
using namespace MathCore::AlgebraCore::MatrixCore;

using namespace MachineLearning;

void SimpleFischerLDA::setFine(std::pair<float, float> _fine)
{
	this->fine = _fine;
};

float SimpleFischerLDA::predict(MathVector<float>& features)
{
	float _prediction_positive = (features * this->alpha_positive) + this->betta_positive + std::log(this->fine.first * this->prioriProbability.first);
	float _prediction_negative = (features * this->alpha_negative) + this->betta_negative + std::log(this->fine.second * this->prioriProbability.second);

	return ((_prediction_positive - _prediction_negative ) > threshold) ? 1. : -1.;
}

void SimpleFischerLDA::learn(std::vector<Instance>& learnSet, std::vector<std::pair<float, float>>& learning_curve)
{
	std::vector<MathVector<float>> *_features = new std::vector<MathVector<float>>();
    _features->reserve(learnSet.size());
    std::for_each(learnSet.begin(), learnSet.end(),
                    [_features](Instance& instance)
                    {
                        _features->push_back(instance.getFeatures());
                    });

	MathMatrix<float> learnF(*_features);

	std::vector<float> learnYValue(learnSet.begin(), learnSet.end());

	MathVector<float> learnY(learnYValue);

    std::cout << "sample means calculating" << std::endl;
	MathMatrix<float> _sample_means = this->sampleMeans(*_features, learnY);

	delete _features;

    std::cout << "calculating covariation" << std::endl;
    MathMatrix<float> _covariation_inverse;
	this->covariation(learnF, learnY, _sample_means, _covariation_inverse);
    std::cout << "inverse covariation" << std::endl;
    _covariation_inverse = !_covariation_inverse;

	MathMatrix<float> sampleMeans_positive(0, _sample_means[0].getSize());
	MathMatrix<float> sampleMeans_negative(0, _sample_means[1].getSize());

	sampleMeans_positive.push_back(_sample_means[0]);
	sampleMeans_negative.push_back(_sample_means[1]);

    MathMatrix<float> _alpha_negative;
    MathMatrix<float> _alpha_positive;
    std::cout << "calculating alpha positive" << std::endl;
	_alpha_positive = _covariation_inverse * ~sampleMeans_positive;

    std::cout << "calculating alpha negative" << std::endl;
	_alpha_negative = _covariation_inverse * ~sampleMeans_negative;

    std::cout << "calculating betta positive" << std::endl;
	this->betta_positive = (-0.5) * (sampleMeans_positive * _alpha_positive).at(0,0);

    std::cout << "calculating betta negative" << std::endl;
	this->betta_negative = (-0.5) * (sampleMeans_negative * _alpha_negative).at(0,0);

    this->alpha_positive = _alpha_positive.row(0);
    this->alpha_negative = _alpha_negative.row(0);

    std::cout << "Simple Fischer LDA parameters:" << std::endl
              << "\t          treshold : " << threshold << std::endl
              << "\tpriori probability : positive - " << prioriProbability.first << " negative - " << prioriProbability.second << std::endl
              << "\t              fine : positive - " << fine.first << " negative - " << fine.second << std::endl
              << "\t betta coefficient : positive - " << betta_positive << " negative - " << betta_negative << std::endl;

	return;
}

MathMatrix<float>& SimpleFischerLDA::sampleMeans(std::vector<MathVector<float>>& learnF, MathVector<float>& learnY)
{
	float prioriProbability_positive = 0.;
	float prioriProbability_negative = 0.;

	float instances_positive = 0;
	float instances_negative = 0;

	MathVector<float> _sample_means_positive(learnF.front());

	MathVector<float> _sample_means_negative(learnF.front());

	for (size_t index = 1; index < learnF.size(); index++)
	{
		if (learnY.getElement(index) == 1)
		{
			instances_positive++;
			_sample_means_positive = _sample_means_positive + learnF.at(index);
		}
		else
		{
			instances_negative++;
			_sample_means_negative = _sample_means_negative + learnF.at(index);
		}
	}

	prioriProbability_positive = instances_positive / learnF.size();
	prioriProbability_negative = instances_negative / learnF.size();


	prioriProbability = std::make_pair(prioriProbability_positive, prioriProbability_negative);

	_sample_means_positive /= instances_positive;
	_sample_means_negative /= instances_negative;

	std::vector<MathVector<float>> sampleMeans;
	sampleMeans.push_back(_sample_means_positive);
	sampleMeans.push_back(_sample_means_negative);
    HolderVectorNorm<float> euclidean;
	//this->fine = std::make_pair(instances_negative, instances_positive);
    this->fine = std::make_pair(euclidean.calc(_sample_means_positive) * instances_negative, euclidean.calc(_sample_means_negative) * instances_positive);


	MathMatrix<float>* _values = new MathMatrix<float>(sampleMeans);

	return  *_values;
}

void SimpleFischerLDA::covariation(MathMatrix<float>& learnSet, MathVector<float>& learnY, MathMatrix<float>& _sampleMeans, MathMatrix<float>& _covariations)
{
	_covariations = MathMatrix<float>(learnSet.cols_size(), learnSet.cols_size(), 0);
    size_t part = pow(10, 5);
    size_t total_count = 0;
    float regularizeValue = std::pow(10, -5);
    float normalizing = (learnSet.rows_size() - 2);
    float magic = pow(10.0, -8.0);
#pragma omp parallel for
	for (size_t index = 0; index < learnSet.rows_size(); index++)
	{
		MathVector<float> difference;

		if (learnY.getElement(index) == 1)
		{
			difference = learnSet[index] - _sampleMeans[0];
		}
		else
		{
			difference = learnSet[index] - _sampleMeans[1];
		}

		MathVector<float>::fast_iterator outIt = difference.fast_begin();

		for (; outIt != difference.fast_end(); ++outIt)
		{
			MathVector<float>::fast_iterator innerIt = outIt;

			for (; innerIt != difference.fast_end(); ++innerIt)
			{
				float value_1 = (_covariations.at(outIt->first, innerIt->first) + outIt->second * innerIt->second) / normalizing + ((outIt == innerIt) ? regularizeValue : 0.0);
				float value_2 = (_covariations.at(innerIt->first, outIt->first) + outIt->second * innerIt->second) / normalizing;
				#pragma omp atomic
				_covariations.insert_element(value_1, outIt->first, innerIt->first);
                if (outIt != innerIt)
				{
                    #pragma omp atomic
                    _covariations.insert_element(value_2, innerIt->first, outIt->first);
				}
			}
		}

        #pragma omp atomic
        total_count++;
        #pragma omp atomic
        if (total_count % part == 0)
            std::cout << "processed objects: " << total_count << std::endl;
	}

	return;
}
