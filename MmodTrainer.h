#pragma once

#ifndef MMOD_TRAINER
#define MMOD_TRAINER

#include <StdIncludes.h>
#include "MyTrainer.h"
#include "MmodDatasetLoader.h"


template <typename net_type> 
class MmodTrainer : public MyTrainer<net_type>
{
public:
	MmodTrainer(double startingLearningRate,
		string syncFile,
		string outputNetworkFile,
		double minimumLearningRate,
		int iterationWithoutProgressTreshold,
		MmodDatasetLoader* mmodDataLoader) : MyTrainer<net_type>(startingLearningRate, syncFile, outputNetworkFile, minimumLearningRate, iterationWithoutProgressTreshold)
	{
		this->mmodDataLoader = mmodDataLoader;
	}

	~MmodTrainer() {
		delete this->mmodDataLoader;
	}

protected:
	MmodDatasetLoader* mmodDataLoader;
private:
	
	void processSpecificNetTraining() {
		std::vector<matrix<rgb_pixel>> imagesToTrain;
		std::vector<std::vector<mmod_rect>> mmodBoxes;


		this->mmodDataLoader->LoadDatasetPart(imagesToTrain, mmodBoxes);

		if (imagesToTrain.size() == 0) {
			MmodTrainer::mmodDataLoader->resetLoader();
			MmodTrainer::mmodDataLoader->LoadDatasetPart(imagesToTrain, mmodBoxes);
		}

		preprocessTrainingData(imagesToTrain, mmodBoxes);

		MyTrainer::trainer->train_one_step(imagesToTrain, mmodBoxes);
	}

	virtual void preprocessTrainingData(std::vector<matrix<rgb_pixel>>& imagesToTrain, std::vector<std::vector<mmod_rect>>& mmodBoxes) = 0;
};
#endif // MY_TRAINER

