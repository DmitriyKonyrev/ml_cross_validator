#include <algorithm>
#include <vector>
#include <map>
#include <memory>

#include "k_nearest_neighbours.h"

#include "instance.h"
#include "metric.h"
#include "mathvector.h"
#include "mathvector_norm.h"
#include "vp_tree.h"

using namespace DataStructures;
using namespace MathCore::AlgebraCore::VectorCore;
using namespace MathCore::AlgebraCore::VectorCore::VectorNorm;

namespace MachineLearning
{
	KNearestNeighbours::KNearestNeighbours(size_t _featuresCount, std::shared_ptr<MathVectorNorm<float>> distance)
	: Predictor(_featuresCount)
	, m_distance(distance)
	{ }

	float KNearestNeighbours::predict(MathVector<float>& features)
	{
		return predictRaw(Instance(features, 0.0));
	}

	void KNearestNeighbours::learn(std::vector<Instance>& learnSet, std::vector<std::pair<float, float>>& learning_curve)
	{

		size_t positive_count = 0;
		size_t negative_count = 0;
		neighbours_matrix_t neigbours;
        createNeighboursMatrix(neigbours, learnSet);
		for (Instance& instance: learnSet)
		{
			if (instance.getGoal() == 1.0)
				positive_count++;
			else
				negative_count++;
		}

		std::cout << "Start learning" << std::endl;

		size_t maximal_count = std::min(positive_count, negative_count);
		float maximal_count_quality = 0.0;

		size_t minimal_count = 3;
		float minimal_count_quality = 0.0;
		size_t iterations = 0;
		while (abs(maximal_count - minimal_count) > 2)
		{
			size_t iter_count = (maximal_count + minimal_count) / 2;
			if (iter_count % 2 == 0)
				iter_count += 1;
			size_t iter_count_left = iter_count - 2;
			size_t iter_count_right = iter_count + 2;

			std::pair<float, float> qualities = testRaw(neigbours, learnSet, iter_count_left, iter_count_right);
			float left_quality = qualities.first;
			float right_quality = qualities.second;

			if (left_quality < right_quality)
			{
				minimal_count = iter_count;
				minimal_count_quality = left_quality;
			}
			else
			{
				maximal_count = iter_count;
				maximal_count_quality = right_quality;
			}
			iterations++;
			std::cout << "    Iteration: " << iterations                                      << std::endl
					  << "Minimal count: " << minimal_count << " : " << minimal_count_quality << std::endl
					  << "Maximal count: " << maximal_count << " : " << maximal_count_quality << std::endl;
		}

		if (minimal_count_quality > maximal_count_quality)
		{
			m_effective_count = minimal_count;
		}
		else
		{
			m_effective_count = maximal_count;
		}

		auto distance = [this](const Instance& l, const Instance& r) -> double { return calcDist(l, r); };
		std::cout << "Build VP Tree" << std::endl;
		m_neighbours = VpTree<Instance>(distance);
		m_neighbours.create(learnSet);

		std::cout << "          Model complexity: " << get_model_complexity() << std::endl
			      << "Effective neighbours count: " << m_effective_count      << std::endl;
	}

	size_t KNearestNeighbours::get_model_complexity()
	{
		return m_neighbours.size() * (featuresCount + 1);  
	}
	
	void KNearestNeighbours::createNeighboursMatrix( neighbours_matrix_t& neigbours
			                                       , std::vector<Instance>& learnSet)
	{
		std::cout << "calculate distance matrix" << std::endl;
		neigbours = neighbours_matrix_t(learnSet.size(), neighbours_t(learnSet.size() - 1, neighbour_t(0.0, 0.0)));
		for (size_t index_1 = 0; index_1 < learnSet.size(); ++index_1)
		{
#pragma omp parallel for
			for (size_t index_2 = index_1 + 1; index_2 < learnSet.size(); ++index_2)
			{
				float dist = calcDist(learnSet[index_1], learnSet[index_2]);
				neigbours[index_1][index_2 - 1] = neighbour_t(dist, learnSet[index_2].getGoal());
				neigbours[index_2][index_1]     = neighbour_t(dist, learnSet[index_1].getGoal());
			}
		}
		std::cout << "sort negibours data" << std::endl;
#pragma omp parallel for
		for (neighbours_t& neighbour: neigbours)
		{
			std::sort(neighbour.begin(), neighbour.end());
		}
	}


	std::pair<float, float> KNearestNeighbours::predictRawest( neighbours_matrix_t& neigbours
			                                                 , size_t object_index
										                     , size_t left_count
										                     , size_t right_count)
	{
			size_t left_positive_count = 0;
			size_t left_negative_count = 0;

			size_t right_positive_count = 0;
			size_t right_negative_count = 0;
			
			size_t max_count = std::max(left_count, right_count);

			for (size_t k = 0; k < max_count; ++k)
			{
				if (k < left_count)
				{
					if (neigbours[object_index][k].second == 1.0)
					{
						left_positive_count++;
					}
					else
					{
						left_negative_count++;
					}
				}

				if (k < right_count)
				{
					if (neigbours[object_index][k].second == 1.0)
					{
						right_positive_count++;
					}
					else
					{
						right_negative_count++;
					}
				}
			}
			
			float left_prediction  = left_positive_count  > left_negative_count  ? 1.0 : -1.0;
			float right_prediction = right_positive_count > right_negative_count ? 1.0 : -1.0;

			return std::make_pair(left_prediction, right_prediction);
	}

	std::pair<float, float> KNearestNeighbours::testRaw( neighbours_matrix_t& neigbours
													   , std::vector<Instance>& learnSet
								                       , size_t left_count
				                                       , size_t right_count)
	{
		float left_true_positive  = 0.;
		float left_false_positive = 0.;
		float left_true_negative  = 0.;
		float left_false_negative = 0.;

		float right_true_positive  = 0.;
		float right_false_positive = 0.;
		float right_true_negative  = 0.;
		float right_false_negative = 0.;

		auto increment_metrics = [&learnSet]( float prediction
				                            , size_t index
								            , float& true_positive
								            , float& false_positive
				                            , float& true_negative
				                            , float& false_negative)
		{
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
		};

		#pragma omp parallel for reduction (+:right_true_positive,right_false_positive,right_true_negative,right_false_negative,right_true_positive,right_false_positive,right_true_negative,right_false_negative)
		for (size_t index = 0; index < learnSet.size(); index++)
		{

			std::pair<float, float> prediction = predictRawest(neigbours, index, left_count, right_count);
			increment_metrics(prediction.first, index, left_true_positive,  left_false_positive,  left_true_negative,  left_false_negative);
			increment_metrics(prediction.second, index, right_true_positive, right_false_positive, right_true_negative, right_false_negative);
		}
		
		float left_quality =  Metrics::F1ScoreMetric(left_true_positive, left_false_positive, left_true_negative, left_false_negative);
		float right_quality =  Metrics::F1ScoreMetric(right_true_positive, right_false_positive, right_true_negative, right_false_negative);
		return std::make_pair(left_quality, right_quality);
	}

	float KNearestNeighbours::predictRaw(const Instance& object)
	{
		std::vector<Instance>* results = new std::vector<Instance>;
		std::vector<double>* distances = new std::vector<double>();
	    m_neighbours.search(object, (int)m_effective_count, results, distances);
		size_t positive_count = 0;
		size_t negative_count = 0;
		std::for_each(results->begin(), results->end(),
				[&positive_count, &negative_count](const auto& neighbour)
				{
					if (neighbour.getGoal() == 1.0)
						positive_count += 1;
					else
						negative_count += 1;
				});
		delete distances;
		float prediction =  positive_count > negative_count ? 1.0 : -1.0;
		return prediction;
	}

	double KNearestNeighbours::calcDist(const Instance& first, const Instance& second)
	{
		return m_distance->calc(first.getFeatures(), second.getFeatures());
	}
}
