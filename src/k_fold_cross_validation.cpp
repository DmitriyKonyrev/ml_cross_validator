#include <algorithm>
#include <fstream>
#include <vector>
#include <string.h>
#include <algorithm>
#include <ctime>
#include <iostream>

#include <boost/filesystem.hpp>

#ifndef POOL_H
#include "pool.h"
#endif

#ifndef INSTANCE_H
#include "instance.h"
#endif

#ifndef PREDICTOR_H
#include "predictor.h"
#endif

#include <ctime>


#include "k_fold_cross_validation.h"

using namespace MachineLearning;

unsigned int CrossValidation::genRand()
{
	unsigned int result = rand();

	result = result * RAND_MAX + rand();
	result = result * RAND_MAX + rand();
	result = result * RAND_MAX + rand();
	result = result * RAND_MAX + rand();

	return result;
}

unsigned int CrossValidation::genRandLimited(unsigned int limit)
{
	return genRand() % limit;
}

static std::pair<double, double> run(Predictor& _predictor, std::vector<Instance>& _pool, size_t foldsCount)
{
	return std::make_pair(0, 0);
}

std::pair<double, double> CrossValidation::test( Predictor* _predictor
                                             , Pool& _pool
                                             , size_t foldsCount
                                             , std::string testing_category_name
                                             , const std::string& outdir
                                             , bool print)
{
	std::srand(unsigned(std::time(NULL)));

	std::vector<Instance> finalLearnSet = _pool.getInstance();

    boost::filesystem::path output_dir(outdir);
    boost::filesystem::path result_dir(testing_category_name);
    result_dir = output_dir / result_dir;

    boost::filesystem::create_directory(result_dir.string());

    boost::filesystem::path learn_precision_path("learn_precision_path");
    learn_precision_path = result_dir / learn_precision_path;
    std::ofstream learn_precision_path_file;
	learn_precision_path_file.open(learn_precision_path.string());
    boost::filesystem::path test_precision_path("test_precision_path");
    test_precision_path = result_dir / test_precision_path;
    std::ofstream test_precision_path_file;
	test_precision_path_file.open(test_precision_path.string());

    boost::filesystem::path learn_complete_path("learn_complete_path");
    learn_complete_path = result_dir / learn_complete_path;
    std::ofstream learn_complete_path_file;
	learn_complete_path_file.open(learn_complete_path.string());
    boost::filesystem::path test_complete_path("test_complete_path");
    test_complete_path = result_dir / test_complete_path;
    std::ofstream test_complete_path_file;
	test_complete_path_file.open(test_complete_path.string());

    boost::filesystem::path learn_f1_path("learn_f1_path");
    learn_f1_path = result_dir / learn_f1_path;
    std::ofstream learn_f1_path_file;
	learn_f1_path_file.open(learn_f1_path.string());
    boost::filesystem::path test_f1_path("test_f1_path");
    test_f1_path = result_dir / test_f1_path;
    std::ofstream test_f1_path_file;
	test_f1_path_file.open(test_f1_path.string());

    boost::filesystem::path learn_accuracy_path("learn_accuracy_path");
    learn_accuracy_path = result_dir / learn_accuracy_path;
    std::ofstream learn_accuracy_path_file;
	learn_accuracy_path_file.open(learn_accuracy_path.string());
    boost::filesystem::path test_accuracy_path("test_accuracy_path");
    test_accuracy_path = result_dir / test_accuracy_path;
    std::ofstream test_accuracy_path_file;
	test_accuracy_path_file.open(test_accuracy_path.string());

    boost::filesystem::path learn_rmse_path("learn_rmse_path");
    learn_rmse_path = result_dir / learn_rmse_path;
    std::ofstream learn_rmse_path_file;
	learn_rmse_path_file.open(learn_rmse_path.string());
    boost::filesystem::path test_rmse_path("test_rmse_path");
    test_rmse_path = result_dir / test_rmse_path;
    std::ofstream test_rmse_path_file;
	test_rmse_path_file.open(test_rmse_path.string());

    boost::filesystem::path learning_logloss_path("learning_logloss");
    learning_logloss_path = result_dir / learning_logloss_path;
    std::ofstream learning_logloss_path_file;
	learning_logloss_path_file.open(learning_logloss_path.string());

    boost::filesystem::path learning_rmse_path("learning_rmse");
    learning_rmse_path = result_dir / learning_rmse_path;
    std::ofstream learning_rmse_path_file;
	learning_rmse_path_file.open(learning_rmse_path.string());

    boost::filesystem::path time_path("time_path");
    time_path = result_dir / time_path;
    std::ofstream time_path_file;
	time_path_file.open(time_path.string());

    boost::filesystem::path model_complexity_path("model_complexity_path");
    model_complexity_path = result_dir / model_complexity_path;
    std::ofstream model_complexity_path_file;
	model_complexity_path_file.open(model_complexity_path.string());

	std::vector<std::vector<Instance>> learnSet;
	std::vector<std::vector<Instance>> testSet;

	learnSet.resize(foldsCount);
	testSet.resize(foldsCount);

	std::vector<int> instanceNumbers;

	size_t instancesCount = _pool.getInstanceCount();

	for (int index = 0; index < instancesCount; ++index)
	{
		instanceNumbers.push_back(index);
	}

	std::random_shuffle(instanceNumbers.begin(), instanceNumbers.end());

	for (size_t instanceNumber = 0; instanceNumber < instancesCount; ++instanceNumber)
	{
		size_t learnFoldNumber = instanceNumbers[instanceNumber] % foldsCount;

		for (size_t foldNumber = 0; foldNumber < foldsCount; ++foldNumber)
		{
			(foldNumber == learnFoldNumber ? learnSet[foldNumber] : testSet[foldNumber]).push_back(finalLearnSet.at(instanceNumber));
		}
	}

	if (print)
	{
		std::cout << "Instances are shuffled" << std::endl;
	}

    double average_learn_precision = 0.0;
    double average_learn_complete  = 0.0;
    double average_learn_f1        = 0.0;
    double average_learn_accuracy  = 0.0;
    double average_learn_rmse      = 0.0;

    double average_test_precision = 0.0;
    double average_test_complete  = 0.0;
    double average_test_f1        = 0.0;
    double average_test_accuracy  = 0.0;
    double average_test_rmse      = 0.0;

    double averageDuration = 0.;
	double averageComplexity = 0.;

	std::vector<Metrics::Metric> metrics_vector;
	metrics_vector.push_back(Metrics::PrecisionMetric);
	metrics_vector.push_back(Metrics::RecallMetric);
	metrics_vector.push_back(Metrics::F1ScoreMetric);
    metrics_vector.push_back(Metrics::AccuracyMetric);

    std::vector<std::pair<double, double>> learning_curve;

	for (size_t foldNumber = 0; foldNumber < foldsCount; ++foldNumber)
	{
        learning_curve.clear();
		std::vector<double> objWeights;
		if (print)
		{
			std::cout << "----------------------------------------------------------" << std::endl;
			std::cout << "Fold index " << foldNumber << std::endl;
		}

		double duration = 0.;
		clock_t start, finish;
        std::cout << "Learn set size: " << learnSet.at(foldNumber).size() << std::endl;
		start = clock();
		_predictor->learn(learnSet.at(foldNumber), objWeights, learning_curve);
		finish = clock();

		duration = (double)(finish - start);

		averageDuration += duration;
		averageComplexity += _predictor->get_model_complexity();
        std::cout << "Check learn set" << std::endl;
		std::vector<double> learnCharacteristics = _predictor->test(learnSet.at(foldNumber), metrics_vector);
        double learn_precision = learnCharacteristics.at(0);
        double learn_complete  = learnCharacteristics.at(1);
        double learn_f1        = learnCharacteristics.at(2);
        double learn_accuracy  = learnCharacteristics.at(3);
        double learn_rmse      = learnCharacteristics.at(4);

        average_learn_precision += learn_precision;
        average_learn_complete  += learn_complete;
        average_learn_f1        += learn_f1;
        average_learn_accuracy  += learn_accuracy;
        average_learn_rmse      += learn_rmse;

        learn_precision_path_file << learn_precision << std::endl;
        learn_complete_path_file  << learn_complete  << std::endl;
        learn_f1_path_file        << learn_f1        << std::endl;
        learn_accuracy_path_file  << learn_accuracy  << std::endl;
        learn_rmse_path_file      << learn_rmse      << std::endl;

        std::cout << "Check test set" << std::endl;
		std::vector<double> testCharacteristics = _predictor->test(testSet.at(foldNumber), metrics_vector);
        double test_precision = testCharacteristics.at(0);
        double test_complete  = testCharacteristics.at(1);
        double test_f1        = testCharacteristics.at(2);
        double test_accuracy  = testCharacteristics.at(3);
        double test_rmse      = testCharacteristics.at(4);

        average_test_precision += test_precision;
        average_test_complete  += test_complete;
        average_test_f1        += test_f1;
        average_test_accuracy  += test_accuracy;
        average_test_rmse      += test_rmse;

        test_precision_path_file << test_precision << std::endl;
        test_complete_path_file  << test_complete  << std::endl;
        test_f1_path_file        << test_f1        << std::endl;
        test_accuracy_path_file  << test_accuracy  << std::endl;
        test_rmse_path_file      << test_rmse      << std::endl;

        time_path_file << duration << std::endl;
		model_complexity_path_file << _predictor->get_model_complexity() << std::endl;

        std::cout << "learning time   : " << duration << std::endl;
		std::cout << "model complexity: " << _predictor->get_model_complexity() << std::endl;

        std::cout << "precision  : learn - " << learn_precision << " test - " << test_precision << std::endl;
        std::cout << "completness: learn - " << learn_complete  << " test - " << test_complete << std::endl;
        std::cout << "f1_score   : learn - " << learn_f1        << " test - " << test_f1       << std::endl;
        std::cout << "accuracy   : learn - " << learn_accuracy  << " test - " << test_accuracy << std::endl;
        std::cout << "rmse       : learn - " << learn_rmse      << " test - " << test_rmse     << std::endl;

        std::for_each(learning_curve.begin(), learning_curve.end(),
        [&learning_logloss_path_file, &learning_rmse_path_file](std::pair<double, double>& values)
        {
            learning_logloss_path_file  << values.first << std::endl;
			learning_rmse_path_file << values.second << std::endl;
        });

		std::srand(unsigned(std::time(NULL)));
	}

    average_learn_precision /= foldsCount;
    average_learn_complete  /= foldsCount;
    average_learn_f1        /= foldsCount;
    average_learn_accuracy  /= foldsCount;
    average_learn_rmse      /= foldsCount;

    average_test_precision /= foldsCount;
    average_test_complete  /= foldsCount;
    average_test_f1        /= foldsCount;
    average_test_accuracy  /= foldsCount;
    average_test_rmse      /= foldsCount;


    learn_precision_path_file << average_learn_precision << std::endl;
    learn_complete_path_file  << average_learn_complete  << std::endl;
    learn_f1_path_file        << average_learn_f1        << std::endl;
    learn_accuracy_path_file  << average_learn_accuracy  << std::endl;
    learn_rmse_path_file      << average_learn_rmse      << std::endl;

    test_precision_path_file << average_test_precision << std::endl;
    test_complete_path_file  << average_test_complete  << std::endl;
    test_f1_path_file        << average_test_f1        << std::endl;
    test_accuracy_path_file  << average_test_accuracy  << std::endl;
    test_rmse_path_file      << average_test_rmse      << std::endl;

    averageDuration   /= foldsCount;
	averageComplexity /= foldsCount;

    time_path_file << averageDuration << std::endl;
	model_complexity_path_file << averageComplexity << std::endl;
    std::cout << "K Fold CV Total:" << std::endl;
    std::cout << "\taverage learning time   : " << averageDuration   << std::endl;
	std::cout << "\taverage model complexity: " << averageComplexity << std::endl;

    std::cout << "\taverage precision  : learn - " << average_learn_precision << " test - " << average_test_precision << std::endl;
    std::cout << "\taverage completness: learn - " << average_learn_complete  << " test - " << average_test_complete << std::endl;
    std::cout << "\taverage f1_score   : learn - " << average_learn_f1        << " test - " << average_test_f1       << std::endl;
    std::cout << "\taverage accuracy   : learn - " << average_learn_accuracy  << " test - " << average_test_accuracy << std::endl;
    std::cout << "\taverage rmse       : learn - " << average_learn_rmse      << " test - " << average_test_rmse     << std::endl;

	return std::make_pair(average_learn_rmse, average_test_rmse);
}
