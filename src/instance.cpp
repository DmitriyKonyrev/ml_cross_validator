#include <vector>
#include <string.h>

#include "instance.h"
#include "mathvector.h"

using namespace  MathCore::AlgebraCore::VectorCore;

namespace MachineLearning
{

	double& Instance::operator [](int index) const
	{
		return *(new double(features.get().getElement((size_t)index)));
	}

	double Instance::getGoal() const
	{
		return goal;
	}

	MathVector<double>& Instance::getFeatures() const
	{
		return features.get();
	}


	Instance::operator MathVector<double>&()
	{
		return features.get();
	}

	Instance::operator double() const
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
