#include <vector>

#ifndef METRIC_H
#define METRIC_H


namespace MachineLearning
{

	class Metrics
	{
		public:

		typedef float(*Metric)(float true_positive, float false_positive, float true_negative, float false_negative);

		static float RecallMetric(float true_positive, float false_positive, float true_negative, float false_negative)
		{
			return true_positive / (true_positive + false_negative);
		}

		static float PrecisionMetric(float true_positive, float false_positive, float true_negative, float false_negative)
		{
			return true_positive / (true_positive + false_positive);
		}

		static float F1ScoreMetric(float true_positive, float false_positive, float true_negative, float false_negative)
		{
			float precision = PrecisionMetric(true_positive, false_positive, true_negative, false_negative);
			float recall = RecallMetric(true_positive, false_positive, true_negative, false_negative);

			return 2 * precision * recall / (precision + recall);
		}

		static float AccuracyMetric(float true_positive, float false_positive, float true_negative, float false_negative)
		{
			return (true_positive + true_negative) / (true_positive + false_positive + true_negative + false_negative);
		}

	};

}
#endif
