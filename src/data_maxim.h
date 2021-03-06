#ifndef DATA_MAXIM_H
#define DATA_MAXIM_H


#include <vector>
#include <string.h>

#ifndef DATA_H
#include "data.h"
#endif

#ifndef INSTANCE_H
#include "instance.h"
#endif

#include "mathvector.h"

using namespace MathCore::AlgebraCore::VectorCore;

namespace MachineLearning
{
	class DataMaxim : public Data
	{
		public:

			DataMaxim(std::vector<std::string> _categories, MathVector<double>& _features);
			DataMaxim();

			void parseFrom(std::string data, int feature_count);
	};
}

#endif
