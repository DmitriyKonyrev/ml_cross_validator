#include <vector>
#include <math.h>

#ifndef METRIC_H
#define METRIC_H


namespace MachineLearning
{

	class Metrics
	{
		public:

		typedef double(*Metric)(double true_positive, double false_positive, double true_negative, double false_negative);

		static double RecallMetric(double true_positive, double false_positive, double true_negative, double false_negative)
		{
			return true_positive / (true_positive + false_negative);
		}

		static double PrecisionMetric(double true_positive, double false_positive, double true_negative, double false_negative)
		{
			return true_positive / (true_positive + false_positive);
		}

		static double F1ScoreMetric(double true_positive, double false_positive, double true_negative, double false_negative)
		{
			double precision = PrecisionMetric(true_positive, false_positive, true_negative, false_negative);
			double recall = RecallMetric(true_positive, false_positive, true_negative, false_negative);

			return 2 * precision * recall / (precision + recall);
		}

		static double AccuracyMetric(double true_positive, double false_positive, double true_negative, double false_negative)
		{
			return (true_positive + true_negative) / (true_positive + false_positive + true_negative + false_negative);
		}

		static double Logloss(double real_value, double prediction)
		{
			prediction = (prediction + 1.) / 2.;
			if (real_value == -1.0)
			{
				prediction = 1. - prediction;
			}

			return (-1.0) * prediction * log2(prediction);
		}

		static double RMSE(double real_value, double prediction)
		{
			return std::pow(prediction - real_value, 2.0);
		}

	};

}
#endif
