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
		MathVector<float> features;

		static std::string CIPHERS;

	public:

		Data();
		Data(std::vector<std::string> _categories, MathVector<float>& _features);
		Data(std::string _data);

		Instance& toInstance(std::string category, size_t& positive_count, float& blur_factor);
		Instance& toLinearInstance(std::string category, size_t& positive_count, float& blur_factor);
		virtual void parseFrom(std::string data);

		std::vector<std::string> getCategories();

		float maximum();

		size_t featuresSize();
		void completeFeatures(size_t counts);

		float at(size_t index);
	};
};

#endif
