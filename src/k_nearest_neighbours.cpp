#include <algorithm>
#include <vector>
#include <map>
#include <memory>
#include <limits>

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
	KNearestNeighbours::KNearestNeighbours(size_t _featuresCount, std::shared_ptr<MathVectorNorm<double>> distance, neighbour_weight_t neighbour_weight, bool fris_stolp)
	: Predictor(_featuresCount)
	, m_distance(distance)
	, m_neighbour_weight(neighbour_weight)
	, m_fris_stolp(fris_stolp)
	{ }

	double KNearestNeighbours::predict(MathVector<double>& features)
	{
		return predictRaw(Instance(features, 0.0));
	}

	void KNearestNeighbours::learn( std::vector<Instance>& learnSet
			                      , std::vector<double>& objectsWeights
			                      , std::vector<std::pair<double, double>>& learning_curve)
	{
		size_t positive_count = 0;
		size_t negative_count = 0;
		neighbours_matrix_t neigbours;
		std::vector<std::vector<size_t>> renumerator;
		std::vector<size_t> objects_indexes;
        createNeighboursMatrix(neigbours, renumerator, learnSet, learnSet, objects_indexes);
		for (Instance& instance: learnSet)
		{
			if (instance.getGoal() == 1.0)
				positive_count++;
			else
				negative_count++;
		}

		auto distance = [this](const Instance& l, const Instance& r) -> double { return calcDist(l, r); };
		std::cout << "Build VP Tree" << std::endl;
		m_neighbours = VpTree<Instance>(distance);

		if (m_fris_stolp)
		{
			std::cout << "STOLP selection started" << std::endl;
			std::unordered_set<size_t> selected_objects = select_objects(neigbours, renumerator, learnSet);
			std::vector<Instance> selected_instances;
			positive_count = 0;
			negative_count = 0;
			for (const size_t& object_index: selected_objects)
			{
				selected_instances.push_back(learnSet[object_index]);
				objects_indexes.push_back(object_index);
				if (learnSet[object_index] == 1.0)
					positive_count++; 
				else
					negative_count++;
			}

			neigbours.clear();
			renumerator.clear();
			createNeighboursMatrix(neigbours, renumerator, learnSet, selected_instances, objects_indexes);
			renumerator.clear();
			objects_indexes.clear();
			m_neighbours.create(selected_instances);
		}
		else
		{
			renumerator.clear();
			m_neighbours.create(learnSet);
		}

		std::cout << "Start learning" << std::endl;

		size_t maximal_count = std::min(positive_count, negative_count);
		double maximal_count_quality = 0.0;

		size_t minimal_count = 3;
		double minimal_count_quality = 0.0;
		size_t iterations = 0;
		while (abs(maximal_count - minimal_count) > 2)
		{
			size_t iter_count = (maximal_count + minimal_count) / 2;
			if (iter_count % 2 == 0)
				iter_count += 1;
			size_t iter_count_left = iter_count - 2;
			size_t iter_count_right = iter_count + 2;

			std::pair<double, double> qualities = testRaw(neigbours, learnSet, iter_count_left, iter_count_right);
			double left_quality = qualities.first;
			double right_quality = qualities.second;

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


		std::cout << "          Model complexity: " << get_model_complexity() << std::endl
			      << "Effective neighbours count: " << m_effective_count      << std::endl;
	}

	size_t KNearestNeighbours::get_model_complexity()
	{
		return m_neighbours.size() * (featuresCount + 1);  
	}
	
	void KNearestNeighbours::createNeighboursMatrix( neighbours_matrix_t& neigbours
			                                       , std::vector<std::vector<size_t>>& renumerator
			                                       , std::vector<Instance>& learnSet
												   , std::vector<Instance>& objects
												   , std::vector<size_t>& objects_indexes)
	{
		std::cout << "calculate distance matrix" << std::endl;
		if (learnSet.size() == objects.size())
		{
			neigbours = neighbours_matrix_t(learnSet.size(), neighbours_t(learnSet.size() - 1, neighbour_t(0.0, {0, 0.0})));
			renumerator = std::vector<std::vector<size_t>>(learnSet.size(), std::vector<size_t>(learnSet.size(), 0));
		}
		else
		{
			neigbours = neighbours_matrix_t(learnSet.size(), neighbours_t(objects.size(), neighbour_t(0.0, {0, 0.0})));
			renumerator = std::vector<std::vector<size_t>>(learnSet.size(), std::vector<size_t>(objects.size(), 0));
		}

		for (size_t index_1 = 0; index_1 < learnSet.size(); ++index_1)
		{
			if (learnSet.size() == objects.size())
#pragma omp parallel for
				for (size_t index_2 = index_1 + 1; index_2 < learnSet.size(); ++index_2)
				{
					double dist = calcDist(learnSet[index_1], learnSet[index_2]);
					neigbours[index_1][index_2 - 1] = neighbour_t(dist, {index_2, learnSet[index_2].getGoal()});
					neigbours[index_2][index_1]     = neighbour_t(dist, {index_1, learnSet[index_1].getGoal()});
				}
			else
			{
#pragma omp parallel for
				for (size_t index_2 = 0; index_2 < objects.size(); ++index_2)
				{
					double dist = (index_1 == objects_indexes[index_2]) ? std::numeric_limits<double>::max() : calcDist(learnSet[index_1], objects[index_2]);
					neigbours[index_1][index_2] = neighbour_t(dist, {index_1, objects[index_2].getGoal()});
				}
			}
		}
		std::cout << "sort negibours data" << std::endl;
#pragma omp parallel for
		for (size_t neighbour_index = 0; neighbour_index < neigbours.size(); ++neighbour_index)
		{
			std::sort(neigbours[neighbour_index].begin(), neigbours[neighbour_index].end());
			if (learnSet.size() == objects.size())
				for (size_t index = 0; index < neigbours[neighbour_index].size() - 1; ++index)
					renumerator[neighbour_index][neigbours[neighbour_index][index].second.first] = index;
		}
	}


	std::pair<double, double> KNearestNeighbours::predictRawest( neighbours_matrix_t& neigbours
			                                                 , size_t object_index
										                     , size_t left_count
										                     , size_t right_count)
	{
			double left_positive_count = 0;
			double left_negative_count = 0;

			double right_positive_count = 0;
			double right_negative_count = 0;
			
			size_t max_count = std::max(left_count, right_count);

			for (size_t k = 0; k < max_count; ++k)
			{
				if (k < left_count)
				{
					if (neigbours[object_index][k].second.second == 1.0)
					{
						left_positive_count += m_neighbour_weight(k, left_count);
					}
					else
					{
						left_negative_count += m_neighbour_weight(k, left_count);
					}
				}

				if (k < right_count)
				{
					if (neigbours[object_index][k].second.second == 1.0)
					{
						right_positive_count += m_neighbour_weight(k, right_count);
					}
					else
					{
						right_negative_count += m_neighbour_weight(k, right_count);
					}
				}
			}
			
			double left_prediction  = left_positive_count  > left_negative_count  ? 1.0 : -1.0;
			double right_prediction = right_positive_count > right_negative_count ? 1.0 : -1.0;

			return std::make_pair(left_prediction, right_prediction);
	}

	std::pair<double, double> KNearestNeighbours::testRaw( neighbours_matrix_t& neigbours
													   , std::vector<Instance>& learnSet
								                       , size_t left_count
				                                       , size_t right_count)
	{
		double left_true_positive  = 0.;
		double left_false_positive = 0.;
		double left_true_negative  = 0.;
		double left_false_negative = 0.;

		double right_true_positive  = 0.;
		double right_false_positive = 0.;
		double right_true_negative  = 0.;
		double right_false_negative = 0.;

		auto increment_metrics = [&learnSet]( double prediction
				                            , size_t index
								            , double& true_positive
								            , double& false_positive
				                            , double& true_negative
				                            , double& false_negative)
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

			std::pair<double, double> prediction = predictRawest(neigbours, index, left_count, right_count);
			increment_metrics(prediction.first, index, left_true_positive,  left_false_positive,  left_true_negative,  left_false_negative);
			increment_metrics(prediction.second, index, right_true_positive, right_false_positive, right_true_negative, right_false_negative);
		}
		
		double left_quality =  Metrics::F1ScoreMetric(left_true_positive, left_false_positive, left_true_negative, left_false_negative);
		double right_quality =  Metrics::F1ScoreMetric(right_true_positive, right_false_positive, right_true_negative, right_false_negative);
		return std::make_pair(left_quality, right_quality);
	}

	double KNearestNeighbours::predictRaw(const Instance& object)
	{
		std::vector<Instance>* results = new std::vector<Instance>;
		std::vector<double>* distances = new std::vector<double>();
	    m_neighbours.search(object, (int)m_effective_count, results, distances);
		double positive_count = 0;
		double negative_count = 0;
		size_t index = 0;
		std::for_each(results->begin(), results->end(),
				[this, &index, &positive_count, &negative_count](const auto& neighbour)
				{
					if (neighbour.getGoal() == 1.0)
						positive_count += m_neighbour_weight(index, m_effective_count);
					else
						negative_count += m_neighbour_weight(index, m_effective_count);
					index++;
				});
		delete distances;
		double prediction =  positive_count > negative_count ? 1.0 : -1.0;
		return prediction;
	}

	double KNearestNeighbours::calcDist(const Instance& first, const Instance& second)
	{
		return m_distance->calc(first.getFeatures(), second.getFeatures());
	}

	double KNearestNeighbours::fris_stolp( neighbours_matrix_t& neighbours
			                            , std::vector<std::vector<size_t>>& renumerator
										, size_t first
										, size_t second
										, size_t base)
	{
		size_t first_second_index = renumerator[first][second];
		size_t first_base_index   = renumerator[first][base];
		double first_second_dist = neighbours[first][first_second_index].first;
		double first_base_dist   = neighbours[first][first_base_index].first;
		return (-first_second_dist + first_base_dist) / (first_second_dist + first_base_dist) + 1.0;
	}

	size_t KNearestNeighbours::get_nearest_neighbour( neighbours_matrix_t& neighbours
			                                        , std::vector<std::vector<size_t>>& renumerator
			                                        , size_t object_index
													, const std::unordered_set<size_t>&  available_objects)
	{
		size_t object_index_f = 0;
		if (available_objects.size()> 0.4 * neighbours.size())
			for(const neighbour_t& neighbour: neighbours[object_index])
			{
				if (available_objects.find(neighbour.second.first) != available_objects.end())
				{
					object_index_f = neighbour.second.first;
					break;
				}
			}
		else
		{
			double min_dist = neighbours[object_index][renumerator[object_index][*available_objects.begin()]].first;
			for (const size_t& n_index: available_objects)
			{
				if (n_index == object_index)
					continue;
				double dist = neighbours[object_index][renumerator[object_index][n_index]].first;
				if (dist <= min_dist)
				{
					object_index_f = n_index;
					min_dist = dist;
				}
			}
		}

		return object_index_f;
	}

	double KNearestNeighbours::calculate_efficiency( neighbours_matrix_t& neighbours
												  , std::vector<std::vector<size_t>>& renumerator
												  , std::vector<Instance>& objects
												  , size_t object_index
												  , const std::unordered_set<size_t>&  positive_objects
												  , const std::unordered_set<size_t>&  negative_objects
												  , const std::unordered_set<size_t>&  ethalons)
	{
		double defense = 0.0;
#pragma omp parallel for reduction (+:defense)
		for(const size_t& object: positive_objects)
		{
			if (object_index == object)
				continue;
			size_t nearest = get_nearest_neighbour(neighbours, renumerator, object, ethalons);
			double fris = fris_stolp(neighbours, renumerator, object, object_index, nearest);
			defense = defense + fris;
		}
		if (positive_objects.size() > 1)
			defense /= (double)(positive_objects.size() - 1);
		double tolerance = 0.0;
#pragma omp parallel for reduction (+:tolerance)
		for(const size_t& object: negative_objects)
		{
			size_t nearest = get_nearest_neighbour(neighbours, renumerator, object, ethalons);
			double fris = fris_stolp(neighbours, renumerator, object, object_index, nearest);
			tolerance = tolerance + fris;
		}
		if (!negative_objects.empty())
			tolerance /= (double)negative_objects.size();
		return 0.5 * defense + 0.5 * tolerance;
	}

	std::pair<size_t, size_t> KNearestNeighbours::find_ethalons( neighbours_matrix_t& neighbours
			                                                   , std::vector<std::vector<size_t>>& renumerator
										                       , std::vector<Instance>& objects
										                       , const std::unordered_set<size_t>&  positive_objects
															   , const std::unordered_set<size_t>&  positive_ethalons
										                       , const std::unordered_set<size_t>&  negative_objects
															   , const std::unordered_set<size_t>&  negative_ethalons)
	{
		auto most_efficienct = [this, &neighbours, &renumerator, &objects] ( const std::unordered_set<size_t>&  positive_objects
													                       , const std::unordered_set<size_t>&  negative_objects
															               , const std::unordered_set<size_t>&  ethalons) -> size_t
		{
			size_t most_efficient_positive = neighbours.size();
			double  most_efficiency_positive = -10.0;
			for (const size_t& object_index: positive_objects)
			{
				double efficiency = calculate_efficiency( neighbours
						                               , renumerator
						                               , objects
													   , object_index
													   , positive_objects
													   , negative_objects
													   , ethalons);

				if (efficiency >= most_efficiency_positive)
				{
					most_efficient_positive  = object_index;
					most_efficiency_positive = efficiency;
				}
			}

			return most_efficient_positive;
		};

		size_t ethalon_positive = most_efficienct(positive_objects, negative_objects, negative_ethalons);
		size_t ethalon_negative = most_efficienct(negative_objects, positive_objects, positive_ethalons);
		return std::make_pair(ethalon_positive, ethalon_negative);
	}

	std::unordered_set<size_t> KNearestNeighbours::check_fail_objects( neighbours_matrix_t& neighbours
			                                                         , std::vector<std::vector<size_t>>& renumerator
												                     , std::vector<Instance>& objects
												                     , const std::unordered_set<size_t>&  positive_objects
																	 , const std::unordered_set<size_t>&  positive_ethalons
												                     , const std::unordered_set<size_t>&  negative_objects
																	 , const std::unordered_set<size_t>&  negative_ethalons)
	{
		auto check_failes = [this, &neighbours, &renumerator, &objects]( const std::unordered_set<size_t>&  positive_objects
														               , const std::unordered_set<size_t>&  positive_ethalons 
				                                                       , const std::unordered_set<size_t>&  negative_objects
														               , const std::unordered_set<size_t>&  negative_ethalons)
											   	   -> std::unordered_set<size_t>
		{
			std::vector<std::pair<size_t, double>> objects_fris;
			double max_fris = 0.0;
			size_t max_obj = 0;
			double min_fris = 2.0;
			size_t min_obj = 0;
			for (const size_t& object_index: positive_objects)
			{
				size_t positive_nearest = get_nearest_neighbour(neighbours, renumerator, object_index, positive_ethalons);
				size_t negative_nearest = get_nearest_neighbour(neighbours, renumerator, object_index, negative_ethalons);

				double fris = fris_stolp(neighbours, renumerator, object_index, positive_nearest, negative_nearest);
				if (fris >= max_fris)
				{
					max_fris = fris;
					max_obj = object_index;
				}
				if (fris <= min_fris)
				{
					min_fris = fris;
					min_obj = object_index;
				}
				objects_fris.push_back({object_index, fris});
			}

			double mean_fris = (max_fris - min_fris) / 2;
			std::cout << max_fris << " " << mean_fris << " " << min_fris << " " << objects_fris.size() << std::endl;
			std::unordered_set<size_t> filtered_objects;
			double true_treshold = std::min(1.8, max_fris - 0.05 * mean_fris);
			double fail_treshold = std::max(0.2, 0.05 * mean_fris + min_fris);
			for (const auto& value: objects_fris)
			{
				if (value.second <= fail_treshold || value.second >= true_treshold)
				{
					filtered_objects.insert(value.first);
				}
			}

			//filtered_objects.insert(max_obj);
			//filtered_objects.insert(min_obj);

			return filtered_objects;
		};

		std::unordered_set<size_t> positive_failes = check_failes(positive_objects, positive_ethalons, negative_objects, negative_ethalons);
		std::unordered_set<size_t> negative_failes = check_failes(negative_objects, negative_ethalons, positive_objects, positive_ethalons);
		if (!negative_failes.empty())
			positive_failes.insert(negative_failes.begin(), negative_failes.end());

		return positive_failes;
	}

	std::unordered_set<size_t> KNearestNeighbours::select_objects( neighbours_matrix_t& neighbours
			                                                     , std::vector<std::vector<size_t>>& renumerator
												                 , std::vector<Instance>& objects)
	{
		std::unordered_set<size_t> positive_objects;
		std::unordered_set<size_t> negative_objects;

		std::cout << "split objects by classes" << std::endl;

		for (size_t object_index = 0; object_index < objects.size(); ++object_index)
		{
			if (objects[object_index].getGoal() == 1.0)
				positive_objects.insert(object_index);
			else
				negative_objects.insert(object_index);
		}

		std::cout << "init ethalons" << std::endl;
		std::pair<size_t, size_t> init_ethalons = find_ethalons( neighbours
				                                               , renumerator
				                                               , objects
															   , positive_objects
															   , positive_objects
															   , negative_objects
															   , negative_objects);

		std::unordered_set<size_t> positive_ethalons({init_ethalons.first});
		positive_objects.erase(init_ethalons.first);

		std::unordered_set<size_t> negative_ethalons({init_ethalons.second});
		negative_objects.erase(init_ethalons.second);

		size_t iteration = 0;

		while (!positive_objects.empty() || !negative_objects.empty())
		{
			std::cout << "iteration: " << iteration << std::endl;
			std::cout << "positive objects: "  << positive_objects.size()  << std::endl
				      << "negative objects: "  << negative_objects.size()  << std::endl
					  << "positive ethalons: " << positive_ethalons.size() << std::endl
					  << "negative ethalons: " << negative_ethalons.size() << std::endl;
			std::cout << "\tcheck good classified and noise objects" << std::endl;
			std::unordered_set<size_t> failes = check_fail_objects( neighbours
					                                              , renumerator
					                                              , objects
																  , positive_objects
																  , positive_ethalons
																  , negative_objects
																  , negative_ethalons);

			std::cout << "\t\tcount: " << failes.size() << std::endl;

			for (const size_t& object_index: failes)
			{
				if (objects[object_index].getGoal() == 1.0)
				{
					positive_objects.erase(object_index);
					//positive_ethalons.erase(object_index);
				}
				else
				{
					negative_objects.erase(object_index);
					//negative_ethalons.erase(object_index);
				}
			}
			bool positive_cleaned = positive_ethalons.empty();
			bool negative_cleaned = negative_ethalons.empty();
			std::cout << "find ethalons" << std::endl;
			if (positive_ethalons.empty())
				positive_ethalons = positive_objects;
			if (negative_ethalons.empty())
				negative_ethalons = negative_objects;


			std::pair<size_t, size_t> ethalons = find_ethalons( neighbours
					                                          , renumerator
					                                          , objects
															  , positive_objects
															  , positive_ethalons
															  , negative_objects
															  , negative_ethalons);
			if (ethalons.first != (size_t)-1)
			{
				if (positive_cleaned)
					positive_ethalons.clear();
				positive_ethalons.insert(ethalons.first);
				positive_objects.erase  (ethalons.first);
			}

			if (ethalons.second != (size_t)-1)
			{
				if (negative_cleaned)
					negative_ethalons.clear();
				negative_ethalons.insert(ethalons.second);
				negative_objects.erase  (ethalons.second);
			}

			iteration++;
		}

		positive_ethalons.insert(negative_ethalons.begin(), negative_ethalons.end());
		positive_ethalons.insert(positive_objects.begin(), positive_objects.end());
		positive_ethalons.insert(negative_objects.begin(), negative_objects.end());
		return positive_ethalons;
	}
}
