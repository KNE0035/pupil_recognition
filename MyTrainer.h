#pragma once

#ifndef MY_TRAINER
#define MY_TRAINER

#include <StdIncludes.h>

template <typename net_type> 
class MyTrainer
{
public:
	long trainingTimeMinutes;

	MyTrainer(double startingLearningRate, string syncFile, string outputNetworkFile, double minimumLearningRate, int iterationWithoutProgressTreshold) {
		this->learningRate = startingLearningRate;
		this->outputNetworkFile = outputNetworkFile;
		this->syncFile = syncFile;
		this->iterationWithoutProgressTreshold = iterationWithoutProgressTreshold;
		this->minimumLearningRate = minimumLearningRate;
	}

	void train() {
		auto start = std::chrono::high_resolution_clock::now();
		
		processSpecificNetTraining();
		auto finish = std::chrono::high_resolution_clock::now();
		auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);
		this->trainingTimeMinutes = (microseconds.count() * 10e6) / 60;
	}

	~MyTrainer() {
	}

protected:
	double learningRate;
	double minimumLearningRate;
	int iterationWithoutProgressTreshold;
	string syncFile;
	string outputNetworkFile;
private:
	virtual void processSpecificNetTraining() = 0;
};
#endif // MY_TRAINER

