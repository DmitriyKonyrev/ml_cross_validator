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

			float threshold;

			std::pair<float, float> prioriProbability;

			std::pair<float, float> fine;

			MathVector<float> alpha_positive;
			MathVector<float> alpha_negative;

			float betta_positive;
			float betta_negative;

		public:

			SimpleFischerLDA(size_t _featuresCount)
				: Predictor(_featuresCount), threshold(0)
			{
				fine = std::make_pair(1., 1.);
			};

			void setFine(std::pair<float, float> _fine);

			float predict(MathVector<float>& features);

			void learn(std::vector<Instance>& learnSet, std::vector<std::pair<float, float>>& learning_curve);

		protected:

			MathMatrix<float>& sampleMeans(std::vector<MathVector<float>>& learnF, MathVector<float>& learnY);

			void covariation(MathMatrix<float>& learnSet, MathVector<float>& learnY, MathMatrix<float>& _sampleMeans, MathMatrix<float>& transpose_cov);

			size_t get_model_complexity();
	};
}
#endif

