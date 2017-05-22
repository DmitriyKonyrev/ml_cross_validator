#ifndef WEAK_PREDICTOR_H
#define WEAK_PREDICTOR_H

#include <functional>
#include <vector>
#include <memory>

#include "predictor.h"
#include "instance.h"
#include "metric.h"
#include "loss_function_approximation.h"
#include "activation_function.h"
#include "weight_initializer.h"

#include "mathvector.h"

using namespace MathCore::AlgebraCore::VectorCore;

namespace MachineLearning
{
	typedef std::function<bool (Instance& object)> predicate_t;

	class WeakClassifier : public Predictor
	{
	public:
		enum PurityType { INFO_BENEFIT, MUTUAL, KHI_2, GINI };

	public:
		WeakClassifier( size_t feature_count
				      , PurityType type);

		double predict(MathVector<double>& features);
		void learn( std::vector<Instance>& learnSet
				  , std::vector<double>& objectsWeights
				  , std::vector<std::pair<double, double>>& learning_curve);

		size_t get_model_complexity();
		Predictor* clone() const { return new WeakClassifier(*this);};

	private:
		std::pair<double, double> calc_counts( std::vector<Instance>& objects
				, std::vector<double>& objectsWeights
				, predicate_t predicate);
			double evaluate_impurity( std::vector<Instance>& learnSet
									, std::vector<double>& objectsImportance
									, double value
									, size_t feature_index
									, std::pair<double, double>& totals
									, std::unordered_map<double, double>& evaluated);

		double evaluate_info_benefit( std::pair<double, double>& counts
				                    , std::pair<double, double>& totals);
		double  evaluate_mutual_info( std::pair<double, double>& counts
				                    , std::pair<double, double>& totals);
		double        evaluate_khi_2( std::pair<double, double>& counts
				                    , std::pair<double, double>& totals);
		double         evaluate_gini( std::pair<double, double>& counts
				                    , std::pair<double, double>& totals);

	private:
		PurityType m_type;
		size_t	   m_feature_num;
		double     m_value;
		double     m_impurity;

		double     m_positive_class;
	};
}

#endif //WEAK_PREDICTOR_H
