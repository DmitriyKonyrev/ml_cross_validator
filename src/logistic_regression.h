#ifndef LOGISTIC_REGRESSION_CLASSIFIER_H
#define LOGISTIC_REGRESSION_CLASSIFIER_H

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
	class LogisticRegression : public Predictor
	{
		public:
			enum LearningRateTypes { CONST, DIV, EUCLIDEAN };

		protected:

			MathVector<double> weights;
			double threshold;

			HeavisideActivationFunctionLogistic* activate = new HeavisideActivationFunctionLogistic();
			SigmoidActivationFunction* learningActivate = new SigmoidActivationFunction();
			LogisticLossFunction* approximation = new LogisticLossFunction();

			size_t minimalIterations;
			size_t maximalIterations;

			weight_initializer_t weight_init;
			double tau;
			double default_learning_rate;

			bool do_jogging;
			bool do_auto_precision;
			bool do_early_stop;

			LearningRateTypes learning_rate_type;
		public:

			LogisticRegression( size_t _featuresCount
					          , size_t _minimalIterations
							  , size_t _maximalIterations
							  , weight_initializer_t _weight_init = fill_zeroes
							  , double _tau = 0.0
							  , double _default_lr = 1e-4
							  , bool _do_jogging        = false
							  , bool _do_auto_precision = false
							  , bool _do_early_stop     = false
							  , LearningRateTypes _lr_type = LearningRateTypes::CONST)
			: Predictor(_featuresCount)
            , minimalIterations(_minimalIterations)
            , maximalIterations(_maximalIterations)
	        , weight_init(_weight_init)
		    , tau(_tau)
		    , default_learning_rate(_default_lr)
		    , do_jogging         (_do_jogging)
		    , do_auto_precision  (_do_auto_precision)
		    , do_early_stop      (_do_early_stop)
		    , learning_rate_type (_lr_type)
			{ }

			double predict(MathVector<double>& features);
			void learn( std::vector<Instance>& learnSet
					  , std::vector<double>& objectsWeights
					  , std::vector<std::pair<double, double>>& learning_curve);
			double quality(std::vector<Instance>& testSet);
			void setIterationInterval(size_t _minimalIterations, size_t _maximalIterations);

			size_t get_model_complexity();

			Predictor* clone() const { return new LogisticRegression(*this);};
		private:
			double scalarProduct(MathVector<double>& features);
			double predictRaw(double _scalar);
			MathVector<double>& weightsInit(size_t size);
			void weightsJog();
	};
}

#endif
