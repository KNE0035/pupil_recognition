#pragma once

#ifndef MMOD_TRAINER
#define MMOD_TRAINER

#include <StdIncludes.h>
#include "MyTrainer.h"
#include "MmodDatasetLoader.h"

#include <dlib/data_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_transforms.h>


template <typename net_type> 
class MmodTrainer : public MyTrainer<net_type>
{
public:
	MmodTrainer(double startingLearningRate,
		string syncFile,
		string outputNetworkFile,
		double minimumLearningRate,
		int iterationWithoutProgressTreshold,
		chip_dims* normalizedChipDims,
		int detectorWindowTargerSize,
		int detectorWindowMinTargetSize,
		MmodDatasetLoader* mmodDataLoader) : MyTrainer<net_type>(startingLearningRate, syncFile, outputNetworkFile, minimumLearningRate, iterationWithoutProgressTreshold)
	{
		this->mmodDataLoader = mmodDataLoader;
		this->chipDims = normalizedChipDims;
		this->detectorWindowTargerSize = detectorWindowTargerSize;
		this->detectorWindowMinTargetSize = detectorWindowMinTargetSize;
	}

	~MmodTrainer() {
		delete this->mmodDataLoader;
	}

protected:
	chip_dims* chipDims;
	int detectorWindowTargerSize;
	int detectorWindowMinTargetSize;
private:
	MmodDatasetLoader* mmodDataLoader;

	void processSpecificNetTraining() {
		std::vector<matrix<rgb_pixel>> imagesToTrain;
		std::vector<std::vector<mmod_rect>> mmodBoxes;

		MmodTrainer::mmodDataLoader->LoadDatasetPart(imagesToTrain, mmodBoxes);
		MmodTrainer::mmodDataLoader->resetLoader();
		//to do load all mmodBoxes to options

		mmod_options options(mmodBoxes, this->detectorWindowTargerSize, this->detectorWindowMinTargetSize);

		net_type net(options);
		net.subnet().layer_details().set_num_filters(options.detector_windows.size());
		dnn_trainer<net_type> trainer(net);

		trainer.set_learning_rate(this->learningRate);
		trainer.be_verbose();
		trainer.set_synchronization_file(this->outputNetworkFile, std::chrono::minutes(5));
		trainer.set_iterations_without_progress_threshold(this->iterationWithoutProgressTreshold);

		while (trainer.get_learning_rate() >= this->minimumLearningRate)
		{
			this->mmodDataLoader->LoadDatasetPart(imagesToTrain, mmodBoxes);

			if (imagesToTrain.size() == 0) {
				MmodTrainer::mmodDataLoader->resetLoader();
				MmodTrainer::mmodDataLoader->LoadDatasetPart(imagesToTrain, mmodBoxes);
			}
			
			preprocessTrainingData(imagesToTrain, mmodBoxes);
			trainer.train_one_step(imagesToTrain, mmodBoxes);
		}

		trainer.get_net();

		net.clean();
		serialize(this->outputNetworkFile) << net;
	}

	virtual void preprocessTrainingData(std::vector<matrix<rgb_pixel>>& imagesToTrain, std::vector<std::vector<mmod_rect>>& mmodBoxes) = 0;
};
#endif // MY_TRAINER

