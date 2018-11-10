#pragma once

#ifndef MY_TRAINER
#define MY_TRAINER

#include <StdIncludes.h>
#include <dlib/dnn/core.h>


template <typename net_type> 
class MyTrainer
{
public:
	long trainingTimeMinutes;

	typedef typename net_type::training_label_type training_label_type;
	typedef typename net_type::input_type input_type;

	MyTrainer(double startingLearningRate, string syncFile, string outputNetworkFile, double minimumLearningRate, int iterationWithoutProgressTreshold, bool verboseMode) {
		this->learningRate = startingLearningRate;
		this->outputNetworkFile = outputNetworkFile;
		this->syncFile = syncFile;
		this->iterationWithoutProgressTreshold = iterationWithoutProgressTreshold;
		this->minimumLearningRate = minimumLearningRate;
		this->verboseMode = verboseMode;
	}

	void train() {
		auto start = std::chrono::high_resolution_clock::now();
		
		net_type net = getNetWithSpecificOptions();
		dnn_trainer<net_type> trainer(net);
		trainer.set_learning_rate(this->learningRate);
		
		if (verboseMode) {
			trainer.be_verbose();
		}
		trainer.set_synchronization_file(this->outputNetworkFile, std::chrono::minutes(5));
		trainer.set_iterations_without_progress_threshold(this->iterationWithoutProgressTreshold);

		while (trainer.get_learning_rate() >= this->minimumLearningRate)
		{
			std::vector<input_type> data;
			std::vector<training_label_type> labels;
			
			obtaionNextBatchOfTrainingDataAndLabels(data, labels);

			trainer.train_one_step(data, labels);
		}

		trainer.get_net();
		net.clean();
		serialize(this->outputNetworkFile) << net;

		auto finish = std::chrono::high_resolution_clock::now();
		auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);
		this->trainingTimeMinutes = (microseconds.count() * 10e6) / 60;
	}

protected:
	double learningRate;
	double minimumLearningRate;
	int iterationWithoutProgressTreshold;
	string syncFile;
	string outputNetworkFile;
	bool verboseMode;
private:

	virtual net_type getNetWithSpecificOptions() = 0;
	virtual void obtaionNextBatchOfTrainingDataAndLabels(std::vector<input_type>& data, std::vector<training_label_type>& labels) = 0;
};
#endif // MY_TRAINER

