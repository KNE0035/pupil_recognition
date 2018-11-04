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
		
		*trainer = dnn_trainer<net_type>(this->net);

		this->trainer->set_learning_rate(this->learningRate);
		this->trainer->be_verbose();
		this->trainer->set_synchronization_file(this->outputNetworkFile, std::chrono::minutes(5));
		this->trainer->set_iterations_without_progress_threshold(iterationWithoutProgressTreshold);
	}

	void train() {
		auto start = std::chrono::high_resolution_clock::now();

		while (trainer->get_learning_rate() >= this->minimumLearningRate)
		{
			this->processSpecificNetTraining();
		}

		trainer->get_net();

		net.clean();
		serialize(this->outputNetworkFile) << net;

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
	
	net_type net;
	dnn_trainer<net_type>* trainer;


private:
	virtual void processSpecificNetTraining() = 0;
};
#endif // MY_TRAINER

