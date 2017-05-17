#include <exception>
#include <fstream>
#include <string.h>
#include <iostream>
#include <memory>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include "data_storage.h"
#include"data_storage_maximus.h"

#include "k_fold_cross_validation.h"

#include "predictor.h"
#include "simple_fischer_lda.h"
#include "logistic_regression.h"
#include "k_nearest_neighbours.h"

#include "mathvector_norm.h"

using namespace MathCore::AlgebraCore::VectorCore::VectorNorm;

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
    ("predictor-type,t", boost::program_options::value<std::string>(&predictor_type), "type of predictior (log_regressor, ldf, knn)")
    ;
    boost::program_options::variables_map vm;
	 boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).allow_unregistered().run(), vm);
    boost::program_options::notify(vm);

	//kNN options
	std::string weight_scheme = "const";
	bool do_selecting = false;
	if (predictor_type.compare("knn") == 0)
	{
		desc.add_options()
		("weight-scheme,w", boost::program_options::value<std::string>(&weight_scheme), "scheme of weighting neighbours (const, exp, sigm, hyper, log)")
		("fris-stolp,f", boost::program_options::bool_switch(&do_selecting), "do FRiS-STOLP objects selecting");
	}

    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
    boost::program_options::notify(vm);

    if (vm.count("help"))
    {
        std::cout << "Usage: options_description [options]\n";
        std::cout << desc << std::endl;
        return 0;
    }

	classifier_name += predictor_type;
	if (predictor_type.compare("knn") == 0)
	{
		classifier_name += "_" + weight_scheme;
		if (do_selecting)
			classifier_name += "_fris";
	}

	classifier_name += suffix;
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
        else if (predictor_type.compare("knn") == 0)
		{
			neighbour_weight_t weight;
			if (weight_scheme.compare("const") == 0)
				weight = KNearestNeighbours::const_weight;
			else if (weight_scheme.compare("exp") == 0)
				weight = KNearestNeighbours::exp_weight;
			else if (weight_scheme.compare("sigm") == 0)
				weight = KNearestNeighbours::sigm_weight;
			else if (weight_scheme.compare("hyper") == 0)
				weight = KNearestNeighbours::hyper_weight;
			else
				weight = KNearestNeighbours::log_weight;
			std::shared_ptr<MathVectorNorm<float>> distance(new EuclideanNorm<float>());
			predictor = new KNearestNeighbours(pool.getInstanceCount(), distance, weight, do_selecting);
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
