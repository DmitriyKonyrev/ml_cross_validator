#ifndef SIMPLE_FISCHER_LDA_H
#define SIMPLE_FISCHER_LDA_H

#include "predictor.h"
#include "instance.h"

#include <vector>

#include "mathmatrix.h"
#include "mathvector.h"

using namespace MathCore::AlgebraCore::VectorCore;
using namespace MathCore::AlgebraCore::MatrixCore;

namespace MachineLearning
{
	class SimpleFischerLDA : public Predictor
	{
		protected:

			double threshold;

			std::pair<double, double> prioriProbability;

			std::pair<double, double> fine;

			MathVector<double> alpha_positive;
			MathVector<double> alpha_negative;

			double betta_positive;
			double betta_negative;

		public:

			SimpleFischerLDA(size_t _featuresCount)
				: Predictor(_featuresCount), threshold(0)
			{
				fine = std::make_pair(1., 1.);
			};

			void setFine(std::pair<double, double> _fine);

			double predict(MathVector<double>& features);

			void learn( std::vector<Instance>& learnSet
					  , std::vector<double>& objectsWeights
					  , std::vector<std::pair<double, double>>& learning_curve);

			Predictor* clone() const { return new SimpleFischerLDA(*this);};

		protected:

			MathMatrix<double>& sampleMeans(std::vector<MathVector<double>>& learnF, MathVector<double>& learnY);

			void covariation(MathMatrix<double>& learnSet, MathVector<double>& learnY, MathMatrix<double>& _sampleMeans, MathMatrix<double>& transpose_cov);

			size_t get_model_complexity();
	};
}
#endif

