#ifndef K_NEAREST_NEGIHBOURS
#define K_NEAREST_NEGIHBOURS

#include <vector>
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
		typedef std::pair<float, float> neighbour_t;
		typedef std::vector<neighbour_t> neighbours_t;
		typedef std::vector<neighbours_t> neighbours_matrix_t;

	class KNearestNeighbours : public Predictor
	{
	public:
		KNearestNeighbours(size_t _featuresCount, std::shared_ptr<MathVectorNorm<float>> distance);

		float predict(MathVector<float>& features);
		void learn(std::vector<Instance>& learnSet, std::vector<std::pair<float, float>>& learning_curve);

		size_t getFeaturesCount();
		size_t get_model_complexity();
	
	private:
		void createNeighboursMatrix(neighbours_matrix_t& neigbours, std::vector<Instance>& learnSet);
		std::pair<float, float> predictRawest(neighbours_matrix_t& neigbours, size_t object_index, size_t left_count, size_t right_count);
		std::pair<float, float> testRaw(neighbours_matrix_t& neigbours, std::vector<Instance>& learnSet, size_t left_count, size_t right_count);

		float predictRaw(const Instance& object);
		double calcDist(const Instance& first, const Instance& second);

	private:
		std::shared_ptr<MathVectorNorm<float>> m_distance;
		VpTree<Instance> m_neighbours;
		size_t m_effective_count;
	};
}

#endif //K_NEAREST_NEGIHBOURS
