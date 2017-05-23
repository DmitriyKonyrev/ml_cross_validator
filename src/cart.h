#ifndef CART_H
#define CART_H

#include <vector>
#include <memory>

#include "predictor.h"
#include "instance.h"
#include "metric.h"
#include "weak_predictor.h"
#include "logistic_regression.h"

#include "mathvector.h"

using namespace MathCore::AlgebraCore::VectorCore;

namespace MachineLearning
{
	class AbstractNode
	{
	public:
		AbstractNode( std::shared_ptr<AbstractNode> left
				     , std::shared_ptr<AbstractNode> right
					 , bool is_leaf)
		: m_is_leaf(is_leaf)
		, m_left_child(left)
		, m_right_child(right)
		{ }

		double get_prediction(MathVector<double>& object)
		{
			double prediction = this->get_raw_prediction(object);
			if (m_is_leaf)
				return prediction;
			else if (prediction == -1.0)
				return m_left_child->get_prediction(object);
			else
				return m_right_child->get_prediction(object);
		}

		virtual size_t get_complexity() = 0;

	protected:
		virtual double get_raw_prediction(MathVector<double>& object) = 0;

	protected:
		bool m_is_leaf;
		std::shared_ptr<AbstractNode> m_left_child;
		std::shared_ptr<AbstractNode> m_right_child;
	};

	class PredictorNode : public AbstractNode
	{
	public:
		PredictorNode( PredictorPtr predicate
					 , std::shared_ptr<AbstractNode> left
					 , std::shared_ptr<AbstractNode> right
					 , bool is_leaf = false)
		: AbstractNode(left, right, is_leaf)
		, m_predicate(predicate)
		{ }

		size_t get_complexity() { return m_predicate->get_model_complexity(); } 

	protected:
		double get_raw_prediction(MathVector<double>& object)
		{
			return m_predicate->predict(object);
		}

	protected:
		PredictorPtr m_predicate;	
	};

	class WeakLeaf : public AbstractNode
	{
	public:
		WeakLeaf(double _class)
		: AbstractNode(nullptr, nullptr, true)
		, m_class(_class)
		{} 

		size_t get_complexity() { return 1; }

	protected:
		double get_raw_prediction(MathVector<double>& object)
		{
			return m_class;
		}
	
	private:
		double m_class;
	};

	class DecisionTree : public Predictor
	{
	public:
		DecisionTree( size_t       feature_count
			        , PredictorPtr weak_type
				    , double       quality_max = 0.9
				    , bool         weak_leafed = true
					, PredictorPtr lr_type = nullptr
					, double       pruning_factor = 0.0)
		: Predictor(feature_count)
		, m_weak_type     (weak_type)
		, m_quality_max   (quality_max)
		, m_weak_leafed   (weak_leafed)
		, m_lr_type       (lr_type)
		, m_pruning_factor(pruning_factor)
		{};

		double predict(MathVector<double>& features);
		void learn( std::vector<Instance>& learnSet
				  , std::vector<double>& objectsWeights
				  , std::vector<std::pair<double, double>>& learning_curve);
		
		size_t get_model_complexity();

		Predictor* clone() const { new DecisionTree(*this);};

	private:
		std::shared_ptr<AbstractNode> learn_subtree( std::vector<Instance>& learnSet
												   , std::vector<Instance>& testSet
												   , std::vector<double>& objectsWeights);

	private:
		PredictorPtr m_weak_type;
		double       m_quality_max;

		bool         m_weak_leafed;
		PredictorPtr m_lr_type;
		double       m_pruning_factor;

		std::vector<std::shared_ptr<AbstractNode>> m_tree;
	};
}

#endif //CART_H
