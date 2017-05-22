#include <vector>
#include <limits>
#include <math.h>
#include <memory>
#include <unordered_map>
#include <tuple>

#include "predictor.h"
#include "instance.h"
#include "metric.h"
#include "loss_function_approximation.h"
#include "activation_function.h"
#include "weight_initializer.h"

#include "weak_predictor.h"

#include "mathvector.h"

using namespace MathCore::AlgebraCore::VectorCore;

namespace MachineLearning
{
	WeakClassifier::WeakClassifier( size_t feature_count
								  , PurityType type)
	: Predictor(feature_count)
	, m_type(type)
	{ }

	double WeakClassifier::predict(MathVector<double>& features)
	{
		if (features.getElement(m_feature_num) <= m_value)
			return m_positive_class;
		else
			return -1.0 * m_positive_class;
	}

	void WeakClassifier::learn( std::vector<Instance>& learnSet
							  , std::vector<double>& objectsWeights
							  , std::vector<std::pair<double, double>>& learning_curve)
	{
		if (objectsWeights.empty())
			objectsWeights = std::vector<double>(learnSet.size(), 1.0 / (float)learnSet.size());

		std::vector<double> objectsImportance(learnSet.size(), 0.0);
		for (size_t object_index = 0; object_index < learnSet.size(); ++object_index)
		{
			objectsImportance[object_index] = objectsWeights[object_index] * learnSet.size();
		}

		auto auto_predicate = [](Instance& object) -> bool { return true; };
		std::pair<double, double> total = calc_counts(learnSet, objectsImportance, auto_predicate);

		size_t features_count = learnSet.front().getFeatures().getSize();
		
		double best_impurity = -1.0 * std::numeric_limits<double>::max();
		size_t best_feature  = features_count + 1;
		double beast_value   = 0.0;
#pragma omp parallel for
		for (size_t feature_index = 0; feature_index < features_count; ++feature_index)
		{
			//std::cout << "check feature #" << feature_index << std::endl;
			std::unordered_map<double, double> value_counts;
			std::vector<Instance>::iterator min_value, max_value;
			std::tie(min_value, max_value) = std::minmax_element(learnSet.begin(), learnSet.end(),
			[&feature_index] (Instance const& i1, Instance const& i2)
			{
				return i1.getFeatures().getElement(feature_index) < i2.getFeatures().getElement(feature_index);
			});
			double left_value  = min_value->getFeatures().getElement(feature_index);
			double right_value = max_value->getFeatures().getElement(feature_index);
			double med_value = (right_value - left_value) / 2;
			left_value  -= 1e-3 * med_value;
			right_value += 1e-3 * med_value;

			double phi_factor = (1 + pow(5, 0.5)) / 2.0;
			double precision = (right_value - left_value) / 1e4;
			size_t iterations = 0;
			while (abs(right_value - left_value) > precision)
			{
				double factor = (right_value - left_value) / phi_factor;
				double value_1 = right_value - factor;
				double value_2 = left_value + factor;
				
				double impurity_1 = evaluate_impurity(learnSet, objectsImportance, value_1, feature_index, total, value_counts);
				double impurity_2 = evaluate_impurity(learnSet, objectsImportance, value_2, feature_index, total, value_counts);

				if (impurity_1 <= impurity_2)
				{
					left_value = value_1;
				}
				else
				{
					right_value = value_2;
				}

				iterations++;
				/*std::cout << "\t\t\tleft: " << left_value << " " << impurity_1 << std::endl
					      << "\t\t\tright: " << right_value << " " << impurity_2 << std::endl;*/
			}

			double value = (right_value - left_value) / 2.0;
			double impurity = evaluate_impurity(learnSet, objectsImportance, value, feature_index, total, value_counts);
#pragma omp atomic
			if (impurity > best_impurity)
			{
				best_impurity = impurity;
				best_feature  = feature_index;
				beast_value   = value;
			}

			//std::cout << "\tbest value: " << value << " best impurity: " << impurity << std::endl;
		}

		m_impurity    = best_impurity;
		m_feature_num = best_feature;
		m_value       = beast_value;

		std::vector<Metrics::Metric> metric({Metrics::F1ScoreMetric});
		m_positive_class = 1.0;
		double f1_quality_pos = test(learnSet, metric).front();
		m_positive_class = -1.0;
		double f1_quality_neg = test(learnSet, metric).front();
		if (f1_quality_pos >= f1_quality_neg)
			m_positive_class = 1.0;
		else
			m_positive_class = -1.0;

		std::cout << "Total:  best feature: " << m_feature_num << std::endl
			      << " \t\t  best impurity: " << m_impurity    << std::endl
		          << " \t\t     best value: " << m_value       << std::endl;
		return;
	}

	size_t WeakClassifier::get_model_complexity()
	{
		return 5;
	}

	std::pair<double, double> WeakClassifier::calc_counts( std::vector<Instance>& objects
			                                             , std::vector<double>& objectsImportance
														 , predicate_t predicate)
	{
		double pos_count = 0;
		double neg_count = 0;
#pragma omp parallel for reduction(+:pos_count,neg_count)
		for (size_t object_index = 0; object_index < objects.size(); ++object_index)
		{
			if (predicate(objects[object_index]))
				if (objects[object_index].getGoal() == 1.0)
					pos_count = pos_count + objectsImportance[object_index];
				else
					neg_count = neg_count + objectsImportance[object_index];
		}
	
		pos_count = (pos_count == 0.0) ? 1e-7 : pos_count;
		neg_count = (neg_count == 0.0) ? 1e-7 : neg_count;

		return std::make_pair(pos_count, neg_count);
	}

	double WeakClassifier::evaluate_impurity( std::vector<Instance>& learnSet
											, std::vector<double>& objectsImportance
											, double value
											, size_t feature_index
											, std::pair<double, double>& totals
											, std::unordered_map<double, double>& evaluated)
	{
		auto value_it = evaluated.find(value);
		if (value_it != evaluated.end())
			return value_it->second;
		else
		{
			auto predicate = [&value, &feature_index] (Instance& object)
			{
				return object.getFeatures().getElement(feature_index) <= value;
			};

			std::pair<double, double> counts = calc_counts(learnSet, objectsImportance, predicate);
			double impurity_value = 0.0;
			switch(m_type)
			{
				case PurityType::INFO_BENEFIT:
					impurity_value = evaluate_info_benefit(counts, totals);
					break;
				case PurityType::MUTUAL:
					impurity_value = evaluate_mutual_info(counts, totals);
					break;
				case PurityType::KHI_2:
					impurity_value = evaluate_khi_2(counts, totals);
					break;
				case PurityType::GINI:
					impurity_value = evaluate_gini(counts, totals);
					break;
			}

			evaluated[value] = impurity_value;
			return impurity_value;
		}
	}

	double WeakClassifier::evaluate_info_benefit( std::pair<double, double>& counter
				                                , std::pair<double, double>& total_counter)
	{
			double summary = total_counter.first + total_counter.second;
			double info_categories = (total_counter.first * log2(total_counter.first / summary) +
				                     total_counter.second * log2(total_counter.second / summary)) / summary;

			double feature_summary     = counter.first + counter.second;
			feature_summary = (feature_summary == 0.0) ? 1e-7 : feature_summary;
			double no_feature_summary  = summary - feature_summary;
			no_feature_summary = (no_feature_summary == 0.0) ? 1e-7 : no_feature_summary;
			double no_feature_positive = total_counter.first  - counter.first;
			no_feature_positive = (no_feature_positive == 0.0) ? 1e-7 : no_feature_positive;
			double no_feature_negative = total_counter.second - counter.second;
			no_feature_negative = (no_feature_negative == 0.0) ? 1e-7 : no_feature_negative;

			double info_feature = (counter.first * log2(counter.first / feature_summary) + 
				                  counter.second * log2(counter.second / feature_summary)) / summary;

			double info_no_feature = (no_feature_positive * log2(no_feature_positive / no_feature_summary) +
					                 no_feature_negative * log2(no_feature_negative / no_feature_summary)) / summary;

			return -info_categories + (info_feature + info_no_feature);
	}

	double WeakClassifier::evaluate_mutual_info( std::pair<double, double>& counter
				                               , std::pair<double, double>& total_counter)
	{
#define mutual_info(feature_category, feature, category, count) log2((feature_category * count) / (category * feature)) * (feature_category / feature)
			double summary             = total_counter.first + total_counter.second;
			double feature_summary     = counter.first + counter.second;
			double no_feature_summary  = summary - feature_summary;

			double no_feature_positive = total_counter.first  - counter.first;
			double no_feature_negative = total_counter.second - counter.second;
			feature_summary = (feature_summary == 0.0) ? 1e-7 : feature_summary;
			no_feature_summary = (no_feature_summary == 0.0) ? 1e-7 : no_feature_summary;
			no_feature_positive = (no_feature_positive == 0.0) ? 1e-7 : no_feature_positive;
			no_feature_negative = (no_feature_negative == 0.0) ? 1e-7 : no_feature_negative;

			return mutual_info(counter.first,       feature_summary,    total_counter.first,  summary) +
				   mutual_info(counter.second,      feature_summary,    total_counter.second, summary) +
				   mutual_info(no_feature_positive, no_feature_summary, total_counter.first,  summary) +
                   mutual_info(no_feature_negative, no_feature_summary, total_counter.second, summary);
#undef mutual_info
	}

	double WeakClassifier::evaluate_khi_2( std::pair<double, double>& counter
				                         , std::pair<double, double>& total_counter)
	{
#define khi_2(feature_category, feature, category, count) pow(count * feature_category - feature * category, 2.0) / (feature * category * count)
			double summary             = total_counter.first + total_counter.second;
			double feature_summary     = counter.first + counter.second;
			double no_feature_summary  = summary - feature_summary;
			double no_feature_positive = total_counter.first  - counter.first;
			double no_feature_negative = total_counter.second - counter.second;
			feature_summary = (feature_summary == 0.0) ? 1e-7 : feature_summary;
			no_feature_summary = (no_feature_summary == 0.0) ? 1e-7 : no_feature_summary;
			no_feature_positive = (no_feature_positive == 0.0) ? 1e-7 : no_feature_positive;
			no_feature_negative = (no_feature_negative == 0.0) ? 1e-7 : no_feature_negative;

			return khi_2(counter.first,       feature_summary,    total_counter.first,  summary) +
				   khi_2(counter.second,      feature_summary,    total_counter.second, summary) +
				   khi_2(no_feature_positive, no_feature_summary, total_counter.first,  summary) +
                   khi_2(no_feature_negative, no_feature_summary, total_counter.second, summary);
#undef khi_2
	}
		
	double WeakClassifier::evaluate_gini( std::pair<double, double>& counter
										, std::pair<double, double>& total_counter)
	{
			double summary = total_counter.first + total_counter.second;
			double gini_categories = (pow(total_counter.first, 2.0)  +
									  pow(total_counter.second, 2.0)) / pow(summary, 2.0);

			double feature_summary     = counter.first + counter.second;
			double no_feature_summary  = summary - feature_summary;
			double no_feature_positive = total_counter.first  - counter.first;
			double no_feature_negative = total_counter.second - counter.second;
			feature_summary = (feature_summary == 0.0) ? 1e-7 : feature_summary;
			no_feature_summary = (no_feature_summary == 0.0) ? 1e-7 : no_feature_summary;
			no_feature_positive = (no_feature_positive == 0.0) ? 1e-7 : no_feature_positive;
			no_feature_negative = (no_feature_negative == 0.0) ? 1e-7 : no_feature_negative;

			double gini_feature = (counter.first * counter.first / feature_summary + 
				                   counter.second * counter.second / feature_summary) / summary;

			double gini_no_feature = (no_feature_positive * no_feature_positive / no_feature_summary +
					                  no_feature_negative * no_feature_negative / no_feature_summary) / summary;

			return -gini_categories + (gini_feature + gini_no_feature);
	}
}

