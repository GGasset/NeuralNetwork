#include <stdlib.h>
#include <math.h>

#pragma once
class ValueGeneration
{
public:
	static double GenerateWeight(double minValue, double valueClosestTo0, double maxValue)
	{
		// Sort values to prevent errors
		double maxValueCp = maxValue;
		double minValueCp = minValue;
		maxValue = fmax(minValueCp, maxValueCp);
		minValue = fmin(minValueCp, maxValueCp);

		valueClosestTo0 = fabs(valueClosestTo0);
		bool isMaxValuePositiveAndMinValueNegative = maxValue > 0 && minValue < 0;
		bool areMinAndMaxValuesPositive = minValue >= 0 && maxValue >= 0;
		bool areMinAndMaxValuesNegative = minValue <= 0 && maxValue <= 0;

		double output = 0;

		// output multiplier will determine if output is positive or negative only if maxValue is positive and minValue is negative
		int outputMultiplier = rand() % 2;
		outputMultiplier -= 1 * (outputMultiplier == 0);

		// set output to negative or positive valueClosestTo0 if max value is positive and min value is negative
		output += valueClosestTo0 * (isMaxValuePositiveAndMinValueNegative && (outputMultiplier == 1));
		output -= valueClosestTo0 * (isMaxValuePositiveAndMinValueNegative && (outputMultiplier == -1));

		output += minValue * areMinAndMaxValuesPositive;
		output += maxValue * areMinAndMaxValuesNegative;

		double outputAdditionMultiplier = NextDouble();

		// Set output to its final value if min value is negative and max value positive
		output += (outputAdditionMultiplier * (maxValue - valueClosestTo0)) * (isMaxValuePositiveAndMinValueNegative && outputMultiplier == 1);
		output += (outputAdditionMultiplier * (valueClosestTo0 - minValue)) * (isMaxValuePositiveAndMinValueNegative && outputMultiplier == -1);

		// Set output to its final state if both min/max values are positive or negative
		output += (outputAdditionMultiplier * (maxValue - minValue) * areMinAndMaxValuesPositive);
		output += (outputAdditionMultiplier * (minValue - maxValue) * areMinAndMaxValuesNegative);

		return output;
	}

	static double NextDouble()
	{
		return (rand() % 1000) / 1000.0;
	}
};

