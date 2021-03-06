#ifndef WEIGHT_INITIALIZER_H
#define WEIGHT_INITIALIZER_H

#include <algorithm>
#include <ctime>
#include <functional>
#include <vector>
#include <random>
#include <math.h>
#include <memory>

#include "instance.h"
#include "metric.h"

#include "mathvector.h"
#include "mathvector_norm.h"

using namespace MathCore::AlgebraCore::VectorCore;

namespace MachineLearning
{
	typedef std::function<void(MathVector<double>&, double&, const std::vector<Instance>&)> weight_initializer_t;

	void fill_zeroes( MathVector<double>& weights
			        , double& treshold
					, const std::vector<Instance>& objects);

	void randomize_fill( MathVector<double>& weights
			           , double& treshold
			           , const std::vector<Instance>& objects);


	void info_benefit_filler( MathVector<double>& weights
			                , double& treshold
					        , const std::vector<Instance>& objects);

	void mutual_info_filler( MathVector<double>& weights
			               , double& treshold
					       , const std::vector<Instance>& objects);

	void khi_2_filler( MathVector<double>& weights
			         , double& treshold
					 , const std::vector<Instance>& objects);
}
#endif //WEIGHT_INITIALIZER_H
