#include <vector>
#include <string.h>
#include <unordered_map>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/replace.hpp>

#include "instance.h"
#include "mathvector.h"

#include "data.h"

using namespace MachineLearning;

using namespace MathCore::AlgebraCore::VectorCore;

std::string Data::CIPHERS = "0123456789";

Data::Data()
{
}

Data::Data(std::vector<std::string> _categories, MathVector<float>& _features)
: categories(_categories), features(_features)
{
}

Data::Data(std::string _data)
{
	this->parseFrom(_data);
}

float Data::maximum()
{
	return this->features.getMaximalElement();
}

size_t Data::featuresSize()
{
	return this->features.getSize();
}

void Data::completeFeatures(size_t counts)
{
	this->features.completeWith(counts, 0);

	return;
}

float Data::at(size_t index)
{
	return this->features.getElement(index);
}

Instance& Data::toInstance(std::string category)
{
	float goal = -1;

	if (std::find(this->categories.begin(), this->categories.end(), category) != this->categories.end())
	{
		goal = 1;
	}

	Instance* instance = new Instance(this->features, goal);

	return *instance;
}

Instance& Data::toLinearInstance(std::string category)
{
	float goal = 0;

	if (std::find(this->categories.begin(), this->categories.end(), category) != this->categories.end())
	{
		goal = 1;
	}

	Instance* instance = new Instance(this->features, goal);

	return *instance;
}

void Data::parseFrom(std::string _data)
{
	if (_data.size() == 0)
	{
		return;
	}

	std::vector<std::string> datas;

	boost::split(datas, _data, boost::is_any_of("\t"));

	this->categories.clear();

	this->categories.insert(this->categories.begin(), datas.begin(), datas.end() - 1);

	std::vector<std::string> features;

	boost::split(features, datas.back(), boost::is_any_of(" "));

	std::unordered_map<size_t, float> rawFeatures;
	std::set<size_t> not_nulls;

	for (std::vector<std::string>::iterator it = features.begin(); it != features.end(); ++it)
	{
		size_t position = atoi(it->c_str());

		rawFeatures[position] = 1.;
		not_nulls.insert(position);
	}

	this->features = *(new MathVector<float>(rawFeatures, not_nulls));

	return;
}

std::vector<std::string> Data::getCategories()
{
	return this->categories;
}
