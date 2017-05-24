#include <vector>
#include <memory>
#include <random>
#include <ctime>

#include "predictor.h"
#include "instance.h"
#include "metric.h"
#include "weak_predictor.h"
#include "logistic_regression.h"

#include "mathvector.h"

#include "cart.h"

using namespace MathCore::AlgebraCore::VectorCore;

namespace MachineLearning
{
	double DecisionTree::predict(MathVector<double>& features)
	{
		if (m_tree.empty())
			return 1.0;
		else
			m_tree.back()->get_prediction(features);
	}

	void DecisionTree::learn( std::vector<Instance>& learnSet
							, std::vector<double>& objectsWeights
							, std::vector<std::pair<double, double>>& learning_curve)
	{
		if (objectsWeights.empty())
			objectsWeights = std::vector<double>(learnSet.size(), 1.0 / (double)learnSet.size());
		std::vector<double> learnWeights;
		std::vector<Instance> learnSubset;
		std::vector<Instance> testSubset;
		if (m_pruning_factor > 0.0)
		{
			std::random_device rd;
			std::mt19937 gen(rd());
			std::discrete_distribution<> distribution(objectsWeights.begin(), objectsWeights.end());
			size_t learn_size = learnSet.size() * m_pruning_factor;
			double summary = 0.0;
			for (size_t index = 0; index < learnSet.size(); ++index)
			{
				size_t obj_index = distribution(gen);
				if (index < learn_size)
				{
					learnSubset.push_back(learnSet[obj_index]);
					learnWeights.push_back(objectsWeights[obj_index]);
					summary += objectsWeights[obj_index];
				}
				else
					testSubset.push_back(learnSet[obj_index]);

				for (double& weight: learnWeights)
					weight /= summary;
			}
		}
		else
		{
			learnSubset = learnSet;
			learnWeights = objectsWeights;
		}
		std::cout << "learning start" << std::endl;
		std::shared_ptr<AbstractNode> root = learn_subtree(learnSubset, testSubset, objectsWeights);
		m_tree.push_back(root);
		std::cout << "learning finished" << std::endl
			      << "\t  count of nodes:"  << m_tree.size()          << std::endl
				  << "\tmodel complexity:"  << get_model_complexity() << std::endl;

	}

	size_t DecisionTree::get_model_complexity()
	{
		size_t summ_complexity = 0;
		for (auto& node : m_tree)
			summ_complexity += node->get_complexity();
		return summ_complexity;
	}

	std::shared_ptr<AbstractNode> DecisionTree::learn_subtree( std::vector<Instance>& learnSet
												             , std::vector<Instance>& testSet
												             , std::vector<double>& objectsWeights)
	{
		double positive_factor = 0.0;
		double negative_factor = 0.0;
		size_t positive_count = 0;
		size_t negative_count = 0;
		for (Instance& object: learnSet)
			if (object.getGoal() == 1.0)
				positive_count++;
			else
				negative_count++;

		positive_factor = (double)positive_count / (double)learnSet.size();
		negative_factor = (double)negative_count / (double)learnSet.size();
			
		std::cout << "positive factor: " << positive_factor << " negative factor: " << negative_factor << std::endl;

		if (positive_factor >= 0.95 || negative_factor >= 0.95)
		{
			std::cout << "Weak leaf reached" << std::endl;
			return std::shared_ptr<AbstractNode>(new WeakLeaf((positive_factor > negative_factor) ? 1.0 : -1.0));
		}

		
		std::cout << "Learning node as weak classifier" << std::endl;
		PredictorPtr weak_predicate(m_weak_type->clone());
		std::vector<std::pair<double, double>> learning_curve;
		weak_predicate->learn(learnSet, objectsWeights, learning_curve);
		std::vector<Metrics::Metric> quality_func({Metrics::F1ScoreMetric});
		double current_quality = weak_predicate->test(learnSet, quality_func).front();
		std::cout << "current quality: " << current_quality;
		bool return_leaf = false;
		if (!testSet.empty())
		{
			double test_quality = weak_predicate->test(testSet, quality_func).front();
			if (current_quality - test_quality > 0.05)
				return_leaf = true;
			std::cout << "current test quality" << test_quality;
		}
		std::cout << std::endl;

		if (current_quality >= m_quality_max || std::isnan(current_quality) || return_leaf)
		{
			std::cout << "Strong leaf reached" << std::endl;
			if (std::isnan(current_quality) || m_weak_leafed)
				return std::shared_ptr<AbstractNode>(new WeakLeaf((positive_factor > negative_factor) ? 1.0 : -1.0));
			if (m_lr_type == nullptr)
				return std::shared_ptr<AbstractNode>(new PredictorNode(weak_predicate, nullptr, nullptr, true));
			else
			{
				PredictorPtr lr_predictor(m_lr_type->clone());
				lr_predictor->learn(learnSet, objectsWeights, learning_curve);
				return std::shared_ptr<AbstractNode>(new PredictorNode(lr_predictor, nullptr, nullptr, true));
			}
		}
		else
		{
			std::vector<Instance> leftLearnSubset;
			std::vector<double>   leftWeights;
			std::vector<Instance> leftTestSubset;

			std::vector<Instance> rightLearnSubset;
			std::vector<double>   rightWeights;
			std::vector<Instance> rightTestSubset;

			double left_summary  = 0.0;
			double right_summary = 0.0;
			for (size_t index = 0; index < learnSet.size(); ++index)
			{
				if (weak_predicate->predict(learnSet[index].getFeatures()) == -1.0)
				{
					leftLearnSubset.push_back(learnSet[index]);
					leftWeights.push_back(objectsWeights[index]);
					left_summary += objectsWeights[index];
				}
				else
				{
					rightLearnSubset.push_back(learnSet[index]);
					rightWeights.push_back(objectsWeights[index]);
					right_summary += objectsWeights[index];
				}

			}

			for (double& weight: leftWeights)
				weight /= left_summary;
			for (double& weight: rightWeights)
				weight /= right_summary;

			for (Instance& instance: testSet)
			{
				if (weak_predicate->predict(instance.getFeatures()) == -1.0)
					leftLearnSubset.push_back(instance);
				else
					rightLearnSubset.push_back(instance);
			}

			//if (m_lr_type == nullptr)
			//{
				std::cout << "Learn left subtree" << std::endl;
				std::shared_ptr<AbstractNode> left_node  = learn_subtree(leftLearnSubset,  leftTestSubset,  leftWeights);
				std::cout << "Learn right subtree" << std::endl;
				std::shared_ptr<AbstractNode> right_node = learn_subtree(rightLearnSubset, rightTestSubset, rightWeights);
				m_tree.push_back(left_node);
				m_tree.push_back(right_node);
				return std::shared_ptr<AbstractNode>(new PredictorNode(weak_predicate, left_node, right_node));
			/*}
			else
			{
				std::cout << "LR leaf reached" << std::endl;
				PredictorPtr left_lr (m_lr_type->clone());
				PredictorPtr right_lr(m_lr_type->clone());

				std::cout << "Learn left LR leaf" << std::endl;
				left_lr->learn(leftLearnSubset, leftWeights, learning_curve);
				learning_curve.clear();
				std::cout << "Learn right LR leaf" << std::endl;
				right_lr->learn(rightLearnSubset, rightWeights, learning_curve);

				std::shared_ptr<AbstractNode> left_leaf(new PredictorNode(left_lr, nullptr, nullptr, true));
				std::shared_ptr<AbstractNode> right_leaf(new PredictorNode(right_lr, nullptr, nullptr, true));
				m_tree.push_back(left_leaf);
				m_tree.push_back(right_leaf);
				return std::shared_ptr<AbstractNode>(new PredictorNode(weak_predicate, left_leaf, right_leaf));
			}*/
		}
	}
}
