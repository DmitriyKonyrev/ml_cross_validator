#include <vector>
#include <string.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <stdio.h>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/replace.hpp>

#include "data.h"
#include "data_maxim.h"
#include "pool.h"
#include "data_storage.h"
#include "data_storage_maximus.h"

using namespace MachineLearning;

DataStorageMaximus::DataStorageMaximus(std::vector<Data>& _datas)
: DataStorage(_datas)
{
}

DataStorageMaximus::DataStorageMaximus(std::string fileName)
{
	this->parseFromFile(fileName);
}

void DataStorageMaximus::parseFromFile(std::string fileName)
{
	using namespace std;
	fstream fin;
	fin.open(fileName.c_str());

	double max = 0;
	size_t feature_size = 0;


	if (!fin.is_open())
	{
		throw new std::logic_error("Cannot find file");
	}
	else
	{
		if (!fin.eof())
		{
			string line;
			std::getline(fin, line);

			std::vector<std::string> header_datas;

			boost::split(header_datas, line, boost::is_any_of(" "));

			feature_size = atoi(header_datas.front().c_str());

			for (size_t index = 1; index < header_datas.size(); index++)
			{
				std::vector<std::string> categories_values;

				boost::split(categories_values, header_datas.at(index), boost::is_any_of(":"));

				this->categories.push_back(std::make_pair(categories_values.front(), atoi(categories_values.back().c_str())));
			}

			std::sort(this->categories.begin(), this->categories.end(), predicate());

		}

		while (!fin.eof())
		{
			string line;
			std::getline(fin, line);
			DataMaxim data;
			data.parseFrom(line, feature_size);
			this->datas.push_back(data);

		}
	}


	return;
}
