#include <vector>
#include <string.h>

#include "instance.h"
#include "mathvector.h"

using namespace  MathCore::AlgebraCore::VectorCore;

namespace MachineLearning
{

	float& Instance::operator [](int index) const
	{
		return *(new float(features.get().getElement((size_t)index)));
	}

	float Instance::getGoal() const
	{
		return goal;
	}

	MathVector<float>& Instance::getFeatures() const
	{
		return features.get();
	}


	Instance::operator MathVector<float>&()
	{
		return features.get();
	}

	Instance::operator float() const
	{
		return this->goal;
	}

	size_t Instance::getFeaturesSize()
	{
		return this->features.get().getSize();
	}

	size_t Instance::getNotNullFeaturesSize()
	{
		return this->features.get().getSizeOfNotNullElements();
	}
}
