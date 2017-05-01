#include <vector>
#include <string.h>

#include "instance.h"
#include "math_vector.h"

using namespace  MathCore::AlgebraCore;

namespace MachineLearning
{

	float& Instance::operator [](int index) const
	{
		return *(new float(features.getElement((size_t)index)));
	}

	float Instance::getGoal()
	{
		return goal;
	}

	MathVector<float>& Instance::getFeatures()
	{
		return features;
	}


	Instance::operator MathVector<float>&()
	{
		return features;
	}

	Instance::operator float() const
	{
		return this->goal;
	}


	Instance& Instance::operator = (Instance& instance)
	{
		this->features = instance.getFeatures();

		this->goal = instance.getGoal();

		return *this;
	}

	size_t Instance::getFeaturesSize()
	{
		return this->features.get_dimension();
	}

	size_t Instance::getNotNullFeaturesSize()
	{
		return this->features.getSizeOfNotNullElements();
	}
}
