

#ifndef CROSS_VALIDATION_H
#define CROSS_VALIDATION_H

#include <vector>
#include <string>

#ifndef POOL_H
#include "pool.h"
#endif

#ifndef INSTANCE_H
#include "instance.h"
#endif

#ifndef PREDICTOR_H
#include "predictor.h"
#endif

namespace MachineLearning
{
	class CrossValidation
	{

		public:
			static std::pair<float, float> run(Predictor& _predictor, std::vector<Instance>& _pool, size_t foldsCount);

			static std::pair<float, float> test(Predictor* _predictor, Pool& _pool, size_t folds, std::string testing_category_name, const std::string& outdir, bool print = false);

		private:

			static unsigned int genRand();

			static unsigned int genRandLimited(unsigned int limit);

	};
}

#endif
