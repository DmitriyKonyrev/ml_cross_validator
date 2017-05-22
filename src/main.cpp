#include <exception>
#include <fstream>
#include <string.h>
#include <iostream>
#include <memory>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include "ada_boost.h"
#include "data_storage.h"
#include "data_storage_maximus.h"
#include "k_fold_cross_validation.h"
#include "predictor.h"
#include "simple_fischer_lda.h"
#include "logistic_regression.h"
#include "k_nearest_neighbours.h"
#include "weak_predictor.h"
#include "weight_initializer.h"

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
    ("predictor-type,t", boost::program_options::value<std::string>(&predictor_type), "type of predictior (log_regressor, ldf, knn, weak, adaboost)")
    ;
    boost::program_options::variables_map vm;
	 boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).allow_unregistered().run(), vm);
    boost::program_options::notify(vm);

	//kNN options
	std::string weight_scheme = "const";
	bool do_selecting = false;
	//LR options
	std::string weight_init_type   = "zeros";
	std::string learning_rate_type = "const";
	double learning_rate = 1e-3;
	double regular_factor = 0.0;
	bool weights_jog    = false;
	bool auto_precision = false;
	bool early_stop     = false;
	size_t min_iterations = 0;
	size_t max_iterations = 100;
	//Weak options
	std::string weak_impurity = "gini";
	//AdaBoost options
	std::string estimator_type   = "log_regressor";
	size_t estimators            = 200;
	bool bagging                 = false;
	double bagging_factor         = 1.0;
	std::string quality_criteria = "f1";
	double max_quality            = 0.98;
	bool ensemble_method = predictor_type.compare("adaboost") == 0;
	if (predictor_type.compare("adaboost") == 0)
	{
		desc.add_options()
		("estimator-type,e", boost::program_options::value<std::string>(&estimator_type), "type of estimator-predictior (log_regressor, ldf, knn, weak)")
		("estimators-count"        , boost::program_options::value<size_t>(&estimators), "count of estimators")
		("bagging,b", boost::program_options::bool_switch(&bagging), "do bagging over learn set")
		("bagging-factor" , boost::program_options::value<double>(&bagging_factor)   , "volume of chosen data from learn set")
		("quality-criteria,q", boost::program_options::value<std::string>(&quality_criteria), "criteria of stop-value quality")
		("max-quality" , boost::program_options::value<double>(&max_quality)   , "maximal quality value");
		boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).allow_unregistered().run(), vm);
		boost::program_options::notify(vm);
	}
	if (predictor_type.compare("knn") == 0 || (ensemble_method && estimator_type.compare("knn") == 0))
	{
		desc.add_options()
		("weight-scheme,w", boost::program_options::value<std::string>(&weight_scheme), "scheme of weighting neighbours (const, exp, sigm, hyper, log)")
		("fris-stolp,f", boost::program_options::bool_switch(&do_selecting), "do FRiS-STOLP objects selecting");
	}
	if (predictor_type.compare("log_regressor") == 0 || (ensemble_method && estimator_type.compare("knn") == 0))
	{
		desc.add_options()
		("weight-init,w", boost::program_options::value<std::string>(&weight_init_type), "scheme of weight initialization (zeros, random, info_benefit, khi_2, mutual_info)")
		("learning-rate-type,r", boost::program_options::value<std::string>(&learning_rate_type), "learning rate strategy (const, euclidean, div, optimization)")
		("learning-rate,l" , boost::program_options::value<double>(&learning_rate)   , "base learning rate")
		("regular-factor"  , boost::program_options::value<double>(&regular_factor)  , "regularization factor")
		("weights-jog,j"   , boost::program_options::bool_switch(&weights_jog)      , "do weights jogging")
		("auto-precision,a", boost::program_options::bool_switch(&auto_precision)   , "precision auto calculate")
		("early-stop,s"    , boost::program_options::bool_switch(&early_stop)       , "sg early stopping")
		("min-iter"        , boost::program_options::value<size_t>(&min_iterations) , "min iteration over collection count")
		("max-iter"        , boost::program_options::value<size_t>(&max_iterations) , "max iteration over collection count");
	}
	if (predictor_type.compare("weak") == 0 || (ensemble_method && estimator_type.compare("weak") == 0))
	{
		desc.add_options()
		("weak-impurity,w", boost::program_options::value<std::string>(&weak_impurity), "weak impurity type (info_benefit, khi_2, mutual_info, gini)");
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
    if (predictor_type.compare("adaboost") == 0)
	{
		classifier_name += "_" + std::to_string(estimators);
		if (bagging)
			classifier_name += "_bagging" + std::to_string(bagging_factor);
		classifier_name += "_" + estimator_type;
	}
	if (predictor_type.compare("knn") == 0 || (ensemble_method && estimator_type.compare("knn") == 0))
	{
		classifier_name += "_" + weight_scheme;
		if (do_selecting)
			classifier_name += "_fris";
	}
	if (predictor_type.compare("log_regressor") == 0 || (ensemble_method && estimator_type.compare("log_regressor") == 0))
	{
		classifier_name += "_" + weight_init_type;
		classifier_name += "_" + std::to_string(learning_rate);
		classifier_name += "_" + learning_rate_type;
		classifier_name += "_" + std::to_string(regular_factor);
		if (weights_jog)
			classifier_name += "_jogging";
		if (early_stop)
			classifier_name += "_es";
	}
	if (predictor_type.compare("weak") == 0 || (ensemble_method && estimator_type.compare("weak") == 0))
	{
		classifier_name += "_" + weak_impurity;
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
		double blur_factor = 0.0;
		Pool pool =  storage.toPool(it->first, positive_count, blur_factor);
	    std::cout << "Data are processed: positive count - " << positive_count << " blur factor - " << blur_factor << std::endl;
		dataset_marking_path_file << it->first.c_str() << "\t" << (double)positive_count / (double)pool.getInstanceCount() 
			                                           << "\t" << blur_factor << std::endl;

        Predictor* predictor;
        if (predictor_type.compare("ldf") == 0 || (ensemble_method && estimator_type.compare("ldf") == 0))
        {
            predictor = new SimpleFischerLDA(pool.getInstanceCount());
        }
		if (predictor_type.compare("weak") == 0 || (ensemble_method && estimator_type.compare("weak") == 0))
		{
			WeakClassifier::PurityType purity_type = WeakClassifier::PurityType::INFO_BENEFIT;
			if (weak_impurity.compare("gini") == 0)
				purity_type = WeakClassifier::PurityType::GINI;
			else if (weak_impurity.compare("info_benefit") == 0)
				purity_type = WeakClassifier::PurityType::INFO_BENEFIT;
			else if (weak_impurity.compare("mutual_info") == 0)
				purity_type = WeakClassifier::PurityType::MUTUAL;
			else if (weak_impurity.compare("khi_2") == 0)
				purity_type = WeakClassifier::PurityType::KHI_2;

			predictor = new WeakClassifier(pool.getInstanceCount(), purity_type);
		}
        if (predictor_type.compare("knn") == 0 || (ensemble_method && estimator_type.compare("knn") == 0))
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
			std::shared_ptr<MathVectorNorm<double>> distance(new EuclideanNorm<double>());
			predictor = new KNearestNeighbours(pool.getInstanceCount(), distance, weight, do_selecting);
		}
		if (predictor_type.compare("log_regressor") == 0 || (ensemble_method && estimator_type.compare("log_regressor") == 0))
        {
			weight_initializer_t weight_init = fill_zeroes;
			if (weight_init_type.compare("zeros") == 0)
				weight_init = fill_zeroes;
			else if (weight_init_type.compare("random") == 0)
				weight_init = randomize_fill;
			else if (weight_init_type.compare("info_benefit") == 0)
				weight_init = info_benefit_filler;
			else if (weight_init_type.compare("mutual_info") == 0)
				weight_init = mutual_info_filler;
			else if (weight_init_type.compare("khi_2") == 0)
				weight_init = khi_2_filler;

			LogisticRegression::LearningRateTypes lr_type = LogisticRegression::LearningRateTypes::CONST;
			if (learning_rate_type.compare("const") == 0)
				lr_type = LogisticRegression::LearningRateTypes::CONST;
			else if (learning_rate_type.compare("div") == 0)
				lr_type = LogisticRegression::LearningRateTypes::DIV;
			else if (learning_rate_type.compare("euclidean") == 0)
				lr_type = LogisticRegression::LearningRateTypes::EUCLIDEAN;

    	    predictor = new LogisticRegression( pool.getInstanceCount()
					                          , min_iterations
											  , max_iterations
											  , weight_init
											  , regular_factor
											  , learning_rate
											  , weights_jog
											  , auto_precision
											  , early_stop);
        }

		if (predictor_type.compare("adaboost") == 0)
		{
			Metrics::Metric quality = Metrics::F1ScoreMetric;
			if (quality_criteria.compare("precision") == 0)
				quality = Metrics::PrecisionMetric;
			else if (quality_criteria.compare("recall") == 0)
				quality = Metrics::RecallMetric;
			else if (quality_criteria.compare("f1") == 0)
				quality = Metrics::F1ScoreMetric;
			else if (quality_criteria.compare("accuracy") == 0)
				quality = Metrics::AccuracyMetric;
			PredictorPtr estimator(predictor->clone());
			predictor = new AdaBoost( pool.getInstanceCount()
					                , estimator
									, estimators
									, quality
									, max_quality
									, bagging
									, bagging_factor);
		
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
