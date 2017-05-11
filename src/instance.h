#ifndef INSTANCE_H
#define INSTANCE_H

#include "mathvector.h"

#include <functional>
#include <vector>
#include <string.h>

using namespace MathCore::AlgebraCore::VectorCore;

namespace MachineLearning
{

	class Instance
	{
	private:

		std::reference_wrapper<MathVector<float>> features;
		float goal;

	public:
		Instance(MathVector<float>& _features, float _goal)
			: features(_features), goal(_goal)
		{
		}


		float& operator [](int index) const;

		float getGoal() const;

		MathVector<float>& getFeatures() const;

		operator MathVector<float>&();

		operator float() const;

		size_t getFeaturesSize();

		size_t getNotNullFeaturesSize();

	};
};

#endif
