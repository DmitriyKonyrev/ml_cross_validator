#ifndef K_NEAREST_NEGIHBOURS
#define K_NEAREST_NEGIHBOURS

#include <functional>
#include <unordered_set>
#include <vector>
#include <math.h>
#include <memory>

#include "instance.h"
#include "metric.h"
#include "mathvector.h"
#include "mathvector_norm.h"
#include "predictor.h"
#include "vp_tree.h"

using namespace DataStructures;
using namespace MathCore::AlgebraCore::VectorCore;
using namespace MathCore::AlgebraCore::VectorCore::VectorNorm;

namespace MachineLearning
{
	typedef std::pair<double, std::pair<size_t, double>> neighbour_t;
	typedef std::vector<neighbour_t> neighbours_t;
	typedef std::vector<neighbours_t> neighbours_matrix_t;

	typedef std::function<double(size_t index, size_t k)> neighbour_weight_t;

	class KNearestNeighbours : public Predictor
	{
	public:
		static double const_weight(size_t index, size_t k) { return 1.0; }
		static double hyper_weight(size_t index, size_t k) { return 1.0 / double(k + 1); }
		static double exp_weight  (size_t index, size_t k) { return 0.5 * exp(-0.5 * index); }
		static double sigm_weight (size_t index, size_t k) { return 1.0 - 1.0 / exp(1.0 + exp(-double(index) / double(k))); }
		static double log_weight  (size_t index, size_t k) { return log2(1.0 - exp(-index)); }
		
	public:
		KNearestNeighbours(size_t _featuresCount, std::shared_ptr<MathVectorNorm<double>> distance, neighbour_weight_t neighbour_weight = KNearestNeighbours::const_weight, bool fris_stolp = false);

		double predict(MathVector<double>& features);
		void learn( std::vector<Instance>& learnSet
				  , std::vector<double>& objectsWeights
				  , std::vector<std::pair<double, double>>& learning_curve);

		size_t getFeaturesCount();
		size_t get_model_complexity();

		Predictor* clone() const { return new KNearestNeighbours(*this);};

	private:
		void createNeighboursMatrix( neighbours_matrix_t& neigbours
				                   , std::vector<std::vector<size_t>>& renumerator
				                   , std::vector<Instance>& learnSet
								   , std::vector<Instance>& objects
								   , std::vector<size_t>& objects_indexes);
		std::pair<double, double> predictRawest(neighbours_matrix_t& neigbours, size_t object_index, size_t left_count, size_t right_count);
		std::pair<double, double> testRaw(neighbours_matrix_t& neigbours, std::vector<Instance>& learnSet, size_t left_count, size_t right_count);

		double predictRaw(const Instance& object);
		double calcDist(const Instance& first, const Instance& second);
	double fris_stolp( neighbours_matrix_t& neighbours
			        , std::vector<std::vector<size_t>>& renumerator
					, size_t first
					, size_t second
					, size_t base);


		size_t get_nearest_neighbour( neighbours_matrix_t& neighbours
				                    , std::vector<std::vector<size_t>>& renumerator
			                        , size_t object_index
									, const std::unordered_set<size_t>&  available_objects);

		double calculate_efficiency( neighbours_matrix_t& neighbours
                                  , std::vector<std::vector<size_t>>& renumerator
								  , std::vector<Instance>& objects
								  , size_t object_index
								  , const std::unordered_set<size_t>&  positive_objects
								  , const std::unordered_set<size_t>&  negative_objects
								  , const std::unordered_set<size_t>&  ethalons);

		std::unordered_set<size_t> check_fail_objects( neighbours_matrix_t& neighbours
				                                     , std::vector<std::vector<size_t>>& renumerator
												     , std::vector<Instance>& objects
												     , const std::unordered_set<size_t>&  positive_objects
													 , const std::unordered_set<size_t>&  positive_ethalons
												     , const std::unordered_set<size_t>&  negative_objects
												     , const std::unordered_set<size_t>&  negative_ethalons);


		std::pair<size_t, size_t> find_ethalons( neighbours_matrix_t& neighbours
				                               , std::vector<std::vector<size_t>>& renumerator
											   , std::vector<Instance>& objects
										       , const std::unordered_set<size_t>&  positive_objects
										       , const std::unordered_set<size_t>&  positive_ethalons
										       , const std::unordered_set<size_t>&  negative_objects
										       , const std::unordered_set<size_t>&  negative_ethalons);

		std::unordered_set<size_t> select_objects( neighbours_matrix_t& neighbours
				                                 , std::vector<std::vector<size_t>>& renumerator
												 , std::vector<Instance>& objects);

	private:
		std::shared_ptr<MathVectorNorm<double>> m_distance;
		neighbour_weight_t m_neighbour_weight;
		VpTree<Instance> m_neighbours;
		size_t m_effective_count;
		bool m_fris_stolp;
	};
}

#endif //K_NEAREST_NEGIHBOURS
