#ifndef INSTANCE_H
#define INSTANCE_H

#include "math_vector.h"

#include <vector>
#include <string.h>

using namespace MathCore::AlgebraCore;

namespace MachineLearning
{

	class Instance
	{
	private:

		MathVector<float>& features;
		float goal;

	public:
		Instance(MathVector<float>& _features, float _goal)
			: features(_features), goal(_goal)
		{
		}


		float& operator [](int index) const;

		float getGoal();

		MathVector<float>& getFeatures();

		Instance& operator = (Instance& instance);

		operator MathVector<float>&();

		operator float() const;

		size_t getFeaturesSize();

		size_t getNotNullFeaturesSize();

	};
};

#endif
