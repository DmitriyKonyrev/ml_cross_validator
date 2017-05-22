#include <algorithm>
#include <ctime>
#include <functional>
#include <vector>
#include <random>
#include <math.h>
#include <memory>

#include "instance.h"
#include "metric.h"

#include "mathvector.h"
#include "mathvector_norm.h"

#include "weight_initializer.h"

using namespace MathCore::AlgebraCore::VectorCore;

namespace MachineLearning
{
	void fill_zeroes( MathVector<double>& weights
			        , double& treshold
					, const std::vector<Instance>& objects)
	{
		weights = MathVector<double>(objects.front().getFeatures().getSize(), 0);
		treshold = 0;
	}

	void randomize_fill( MathVector<double>& weights
			           , double& treshold
			           , const std::vector<Instance>& objects)
	{
		size_t size = objects.front().getFeatures().getSize(); 
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<double> distribution(-1 / (double)size, 1 / (double)size);

		std::vector<double> values;

		for (size_t index = 0; index < size; index++)
		{
			values.push_back(distribution(gen));
		}

		treshold = distribution(gen);
		weights.setValues(values);
	}

	void objects_counter( const std::vector<Instance>& objects
			            , std::vector<std::pair<double, double>>& counters
						, std::pair<double, double>& total_counter)
	{
		counters = std::vector<std::pair<double, double>>(objects.front().getFeatures().getSize(), {0.0, 0.0});
		for (const Instance& object: objects)
		{
		    MathVector<double>::const_fast_iterator it  = object.getFeatures().const_fast_begin();
		    MathVector<double>::const_fast_iterator end = object.getFeatures().const_fast_end();
			for (; it != end; ++it)
				if (object.getGoal() == 1.0)
					counters[it.index()].first += 1.0;
				else
					counters[it.index()].second += 1.0;

			if (object.getGoal() == 1.0)
				total_counter.first += 1.0;
			else
				total_counter.second += 1.0;
		}

		for (std::pair<double, double>& counter: counters)
		{
			counter.first  = counter.first == 0.0  ? 1e-6 : counter.first;
			counter.second = counter.second == 0.0 ? 1e-6 : counter.second;
		}
	}

	void calc_stat_values( const std::vector<Instance>& objects
						 , MathVector<double>& weights
						 , double& treshold
						 , std::function<double(std::pair<double, double>&, std::pair<double, double>&)> calculator)
	{
#define normalize(value, max, min) (value - min) / (max - min)
		std::vector<std::pair<double, double>> counters;
		std::pair<double, double>              total_counter;

		objects_counter(objects, counters, total_counter);
		std::vector<double> values;
		values.reserve(counters.size());
		double average = 0.0;
		for (std::pair<double, double>& counter: counters)
		{
			double value = calculator(counter, total_counter);
			average += value;
			values.push_back(value);
		}

		double max_value = *std::max_element(values.begin(), values.end());
		double min_value = *std::min_element(values.begin(), values.end());

#pragma omp parallel for
		for (double& value: values)
			value = normalize(isnan(value) ? min_value : value, max_value, min_value) / 1e2;

		average /= counters.size();
#pragma omp parallel for
		treshold = normalize(average, max_value, min_value) / 1e2;

		weights.setValues(values);
#undef normalize
	}

	void info_benefit_filler( MathVector<double>& weights
			                , double& treshold
					        , const std::vector<Instance>& objects)
	{
		auto calc_info_benefit = []( std::pair<double, double>& counter
				                   , std::pair<double, double>& total_counter)
			-> double
		{
			double summary = total_counter.first + total_counter.second;
			double info_categories = (total_counter.first * log2(total_counter.first / summary) +
				                     total_counter.second * log2(total_counter.second / summary)) / summary;

			double feature_summary     = counter.first + counter.second;
			double no_feature_summary  = summary - feature_summary;
			double no_feature_positive = total_counter.first  - counter.first;
			double no_feature_negative = total_counter.second - counter.second;

			double info_feature = (counter.first * log2(counter.first / feature_summary) + 
				                  counter.second * log2(counter.second / feature_summary)) / summary;

			double info_no_feature = (no_feature_positive * log2(no_feature_positive / no_feature_summary) +
					                 no_feature_negative * log2(no_feature_negative / no_feature_summary)) / summary;

			return -info_categories + (info_feature + info_no_feature);
		};

		calc_stat_values(objects, weights, treshold, calc_info_benefit);
	}

	void mutual_info_filler( MathVector<double>& weights
			               , double& treshold
					       , const std::vector<Instance>& objects)
	{
#define mutual_info(feature_category, feature, category, count) log2((feature_category * count) / (category * feature)) * (feature_category / feature)
		auto calc_mutual_info = []( std::pair<double, double>& counter
				                  , std::pair<double, double>& total_counter)
			-> double
		{
			double summary             = total_counter.first + total_counter.second;
			double feature_summary     = counter.first + counter.second;
			double no_feature_summary  = summary - feature_summary;
			double no_feature_positive = total_counter.first  - counter.first;
			double no_feature_negative = total_counter.second - counter.second;

			return mutual_info(counter.first,       feature_summary,    total_counter.first,  summary) +
				   mutual_info(counter.second,      feature_summary,    total_counter.second, summary) +
				   mutual_info(no_feature_positive, no_feature_summary, total_counter.first,  summary) +
                   mutual_info(no_feature_negative, no_feature_summary, total_counter.second, summary);
		};

		calc_stat_values(objects, weights, treshold, calc_mutual_info);
#undef mutual_info
	}

	void khi_2_filler( MathVector<double>& weights
			         , double& treshold
					 , const std::vector<Instance>& objects)
	{
#define khi_2(feature_category, feature, category, count) pow(count * feature_category - feature * category, 2.0) / (feature * category * count)
		auto calc_khi_2 = []( std::pair<double, double>& counter
				            , std::pair<double, double>& total_counter)
			-> double
		{
			double summary             = total_counter.first + total_counter.second;
			double feature_summary     = counter.first + counter.second;
			double no_feature_summary  = summary - feature_summary;
			double no_feature_positive = total_counter.first  - counter.first;
			double no_feature_negative = total_counter.second - counter.second;

			return khi_2(counter.first,       feature_summary,    total_counter.first,  summary) +
				   khi_2(counter.second,      feature_summary,    total_counter.second, summary) +
				   khi_2(no_feature_positive, no_feature_summary, total_counter.first,  summary) +
                   khi_2(no_feature_negative, no_feature_summary, total_counter.second, summary);
		};

		calc_stat_values(objects, weights, treshold, calc_khi_2);
#undef khi_2
	}
}
