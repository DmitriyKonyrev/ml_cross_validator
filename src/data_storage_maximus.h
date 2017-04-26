#ifndef DATA_STORAGE_MAXIMUS_H
#define DATA_STORAGE_MAXIMUS_H


#ifndef DATA_STORAGE_H
#include "data_storage.h"
#endif

#include <vector>
#include <set>
#include <map>
#include <string.h>
#include <iostream>

#ifndef DATA_H
#include "data.h"
#endif

#ifndef POOL_H
#include "pool.h"
#endif

namespace MachineLearning
{
	class DataStorageMaximus : public DataStorage
	{

		public:

			DataStorageMaximus(std::vector<Data>& _datas);
			DataStorageMaximus(std::string fileName);

			void parseFromFile(std::string fileName);
	};
}

#endif