#include <exception>
#include <fstream>
#include <string.h>
#include <iostream>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include "data_storage.h"
#include"data_storage_maximus.h"

#include "k_fold_cross_validation.h"

#include "predictor.h"
#include "simple_fischer_lda.h"
#include "logistic_regression.h"

#include "loss_function_approximation.h"
#include "activation_function.h"

using namespace MachineLearning;

int main(int argc, char* argv[])
try
{
	std::string classifier_name = "";
    boost::program_options::options_description desc("Validating classficators");
    uint32_t fold_count = 10;
    std::string datafile = "factors.txt";
    std::string outdir = "./";
	std::string suffix = "";
    std::string predictor_type = "log_regressor";
    desc.add_options()
    ("help", "produce help message")
    ("data,d", boost::program_options::value<std::string>(&datafile), "input data file")
    ("output-path,o", boost::program_options::value<std::string>(&outdir), "directory with output files")
	("suffix,s", boost::program_options::value<std::string>(&suffix), "suffix of the output directory")
    ("fold-count,k", boost::program_options::value<uint32_t>(&fold_count), "count of folds to validate")
    ("predictor-type,t", boost::program_options::value<std::string>(&predictor_type), "type of predictior (log_regressor, ldf)")
    ;
    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
    boost::program_options::notify(vm);

    if (vm.count("help"))
    {
        std::cout << "Usage: options_description [options]\n";
        std::cout << desc << std::endl;
        return 0;
    }

	classifier_name += predictor_type + suffix;
    boost::filesystem::path output_dir(outdir);
	boost::filesystem::path classifier_dir(classifier_name);
	output_dir = output_dir / classifier_dir;
	outdir = output_dir.string();
	boost::filesystem::create_directory(outdir);

	boost::filesystem::path dataset_marking_dir("categories_data");
	dataset_marking_dir = output_dir / dataset_marking_dir;
	std::ofstream dataset_marking_path_file;
	dataset_marking_path_file.open(dataset_marking_dir.string());


	std::cout << "k Fold Crossvalidation of " << predictor_type  << " starting..." << std::endl;

	std::cout << "wait while reading data from file \"" << datafile << "\"..."<< std::endl;
	DataStorageMaximus storage(datafile);
	std::cout << "data reading finished" << std::endl;
	std::vector<std::pair<std::string, size_t> > categories = storage.getCategories();

	std::cout << "cv control starting" << std::endl;

    size_t index = 0;
	for (std::vector<std::pair<std::string, size_t> >::iterator it = categories.begin(); it != categories.end(); it++)
	{
        index++;
		std::cout << "Category: " << it->first.c_str() << " | Volume: " << it->second << std::endl;
		size_t positive_count = 0;
		float blur_factor = 0.0;
		Pool pool =  storage.toPool(it->first, positive_count, blur_factor);
	    std::cout << "Data are processed: positive count - " << positive_count << " blur factor - " << blur_factor << std::endl;
		dataset_marking_path_file << it->first.c_str() << "\t" << (float)positive_count / (float)pool.getInstanceCount() 
			                                           << "\t" << blur_factor << std::endl;
        Predictor* predictor;
        if (predictor_type.compare("ldf") == 0)
        {
            predictor = new SimpleFischerLDA(pool.getInstanceCount());
        }
        else
        {
    	    predictor = new LogisticRegression(pool.getInstanceCount(), 3, 100);
        }
	    CrossValidation::test(predictor, pool, fold_count, it->first.c_str(), outdir, true);
	}

	std::cout << "cv control finished" << std::endl;

	return 0;
}
catch (std::exception& e)
{
    std::cout << "cv control failes" << e.what() << std::endl;
}
