#ifndef DATA_H
#define DATA_H


#include <vector>
#include <string.h>

#include "instance.h"
#include "mathvector.h"

using namespace MathCore::AlgebraCore::VectorCore;

namespace MachineLearning
{
	class Data
	{
	protected:

		std::vector<std::string> categories;
		MathVector<double> features;

		static std::string CIPHERS;

	public:

		Data();
		Data(std::vector<std::string> _categories, MathVector<double>& _features);
		Data(std::string _data);

		Instance& toInstance(std::string category, size_t& positive_count, double& blur_factor);
		Instance& toLinearInstance(std::string category, size_t& positive_count, double& blur_factor);
		virtual void parseFrom(std::string data);

		std::vector<std::string> getCategories();

		double maximum();

		size_t featuresSize();
		void completeFeatures(size_t counts);

		double at(size_t index);
	};
};

#endif
