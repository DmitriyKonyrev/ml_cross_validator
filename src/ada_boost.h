#ifndef ADABOOST_H
#define ADABOOST_H

#include <vector>
#include <memory>

#include "predictor.h"
#include "instance.h"
#include "metric.h"

#include "mathvector.h"

using namespace MathCore::AlgebraCore::VectorCore;

namespace MachineLearning
{
	class AdaBoost : public Predictor
	{
	public:
		AdaBoost( size_t _featuresCount
				, PredictorPtr predictor_type
				, size_t max_estimators
				, Metrics::Metric quality_checker = Metrics::F1ScoreMetric
				, double max_quality = 0.98
				, bool bagging = false
				, double bagging_factor = 1.0);

		
		double predict(MathVector<double>& features);
		void learn( std::vector<Instance>& learnSet
				  , std::vector<double>& objectsWeights
				  , std::vector<std::pair<double, double>>& learning_curve);
		
		size_t get_model_complexity();

		Predictor* clone() const { new AdaBoost(*this);};
	private:
		std::vector<PredictorPtr> m_estimators;
		std::vector<double>       m_weights;
		PredictorPtr              m_predictor_type;
		size_t                    m_max_estimators;
		Metrics::Metric           m_quality_checker;
		double                    m_max_quality;
		bool                      m_bagging;
		double                    m_bagging_factor;
	};
}

#endif //ADABOOST_H
