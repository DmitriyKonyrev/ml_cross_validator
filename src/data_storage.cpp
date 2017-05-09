#include <vector>
#include <string.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <stdio.h>
#include <exception>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/replace.hpp>

#include "data.h"
#include "pool.h"
#include "data_storage.h"

#include "mathvector.h"

using namespace MathCore::AlgebraCore::VectorCore;

using namespace MachineLearning;

DataStorage::DataStorage()
{

}

DataStorage::DataStorage(std::vector<Data>& _datas)
	: datas(_datas)
{
}

DataStorage::DataStorage(std::string fileName)
{
	this->parseFromFile(fileName);
}

void DataStorage::parseFromFile(std::string fileName)
{
	using namespace std;
	fstream fin;
	fin.open(fileName.c_str());

	float max = 0;

	std::map < std::string, size_t> _categories;

	if (!fin.is_open())
	{
		throw std::logic_error("Cannot open file");
	}
	else
	{
		while (!fin.eof())
		{
			string line;
			std::getline(fin, line);
			Data data(line);
			this->datas.push_back(data);

			float _max = data.featuresSize();

			if (_max > max)
			{
				max = _max;
			}

			std::vector<std::string> categories = data.getCategories();
			for (size_t index = 0; index < categories.size(); index++)
			{
				if (_categories.find(categories.at(index)) != _categories.end())
				{
					_categories[categories.at(index)] += 1;
				}
				else
				{
					_categories[categories.at(index)] = 1;
				}
			}
		}
	}

	max++;

	for (size_t index = 0; index < this->datas.size(); index++)
	{
		size_t completeToCounts = max - this->datas.at(index).featuresSize();
		this->datas.at(index).completeFeatures(completeToCounts);
	}

	fin.close();

	std::vector<std::pair<std::string, size_t> > _array(_categories.begin(), _categories.end());

	std::sort(_array.begin(), _array.end(), predicate());

	this->categories.clear();
	this->categories = _array;

	return;
}

Pool& DataStorage::toPool(std::string category, size_t& positive_count, float& blur_factor)
{
	std::vector<Instance>* instances = new std::vector<Instance>();

	positive_count = 0;
	blur_factor = 0.0;

	for (int index = 0; index < this->datas.size(); index++)
	{
	  instances->push_back(this->datas.at(index).toInstance(category, positive_count, blur_factor));
	}

	blur_factor /= positive_count;
	Pool* pool = new Pool(*instances);

	delete instances;

	return *pool;
}

Pool& DataStorage::toLinearPool(std::string category, size_t& positive_count, float& blur_factor)
{
	std::vector<Instance>* instances = new std::vector<Instance>();
	positive_count = 0;
	blur_factor = 0.0;

	for (int index = 0; index < this->datas.size(); index++)
	{
		instances->push_back(this->datas.at(index).toLinearInstance(category, positive_count, blur_factor));
	}

	blur_factor /= positive_count;
	Pool* pool = new Pool(*instances);

	delete instances;

	return *pool;
}

std::vector<std::pair<std::string, size_t> > DataStorage::getCategories()
{

	return this->categories;
}

size_t DataStorage::categoriesCount()
{
	return this->categories.size();
}
