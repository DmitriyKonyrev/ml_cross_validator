#ifndef POOL_H
#define POOL_H


#include "instance.h"

#include <vector>
#include <string.h>


namespace MachineLearning
{
	class Pool
	{
	private:

		std::vector<Instance> instances;
		int foldsCount;

	public:

		Pool(std::vector<Instance> _instances);

		void shuffle(std::vector<std::vector<Instance>> &learnSet, std::vector<std::vector<Instance>> &testSet, int foldsCount);

		Instance& getInstanceAt(size_t index);

		std::vector<Instance>& getInstance();

		size_t getInstanceCount();

	
	protected:

		unsigned int genRand();

		unsigned int genRandLimited(unsigned int limit);

	};
};

#endif
