#include <vector>
#include <string.h>
#include <algorithm>
#include <ctime>

#include <boost/random.hpp>
#include <boost/random/variate_generator.hpp>

#ifndef INSTANCE_H
#include "instance.h"
#endif

#include "pool.h"

using namespace MachineLearning;

Instance& Pool::getInstanceAt(size_t index)
{
	if (index > this->instances.size() - 1)
	{
		throw std::out_of_range("Index out of range");
	}

	return this->instances.at(index);
}

std::vector<Instance>& Pool::getInstance()
{
	return this->instances;
}

size_t Pool::getInstanceCount()
{
	return this->instances.size();
}

Pool::Pool (std::vector<Instance> _instances)
		: instances(_instances)
{
	
}

void Pool::shuffle(std::vector<std::vector<Instance>> &learnSet, std::vector<std::vector<Instance>> &testSet, int foldsCount)
{
	std::srand(unsigned(std::time(NULL)));


	learnSet.resize(foldsCount);
	testSet.resize(foldsCount);

	std::vector<int> instanceNumbers;

	for (int i = 0; i < this->instances.size(); ++i)
	{
		instanceNumbers.push_back(i);
	}

	std::random_shuffle(instanceNumbers.begin(), instanceNumbers.end());

	for (size_t instanceNumber = 0; instanceNumber < this->instances.size(); ++instanceNumber)
	{
		size_t testFoldNumber = instanceNumbers[instanceNumber] % foldsCount;

		for (size_t foldNumber = 0; foldNumber < foldsCount; ++foldNumber)
		{
			(foldNumber == testFoldNumber ? testSet[foldNumber] : learnSet[foldNumber]).push_back(this->instances.at(instanceNumber));
		}
	}

	return;
}

unsigned int Pool::genRand()
{
	unsigned int result = rand();

	result = result * RAND_MAX + rand();
	result = result * RAND_MAX + rand();
	result = result * RAND_MAX + rand();
	result = result * RAND_MAX + rand();

	return result;
}

unsigned int Pool::genRandLimited(unsigned int limit)
{
	return genRand() % limit;
}
