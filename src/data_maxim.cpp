#ifndef DATA_MAXIM_H
#include "data_maxim.h"
#endif

#include <vector>
#include <string.h>

#ifndef DATA_H
#include "data.h"
#endif

#ifndef INSTANCE_H
#include "instance.h"
#endif

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/replace.hpp>

#include "math_vector.h"


using namespace MachineLearning;
using namespace MathCore::AlgebraCore;


DataMaxim::DataMaxim(std::vector<std::string> _categories, MathVector<float>& _features)
: Data(_categories, _features)
{
}

DataMaxim::DataMaxim()
{
}


void DataMaxim::parseFrom(std::string _data, int feature_count)
{

	std::map<size_t,float> featuresValue;

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

	for (std::vector<std::string>::iterator it = features.begin(); it != features.end(); ++it)
	{
		std::vector<std::string> feature_values;

		boost::split(feature_values, *it, boost::is_any_of(":"));

		int position = atoi(feature_values.front().c_str());

		featuresValue[position] = atof(feature_values.back().c_str());
	}

	(this->features) = *(new MathVector<float>(featuresValue, feature_count));

	return;
}
