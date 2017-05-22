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

		std::reference_wrapper<MathVector<double>> features;
		double goal;

	public:
		Instance(MathVector<double>& _features, double _goal)
			: features(_features), goal(_goal)
		{
		}


		double& operator [](int index) const;

		double getGoal() const;

		MathVector<double>& getFeatures() const;

		operator MathVector<double>&();

		operator double() const;

		size_t getFeaturesSize();

		size_t getNotNullFeaturesSize();

	};
};

#endif
