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
		int detectorWindowTargetSize,
		int detectorWindowMinTargetSize,
		bool cropperRandomlyFlip,
		int cropperMaxRotationDegrees,
		bool verboseMode,
		MmodDatasetLoader* mmodDataLoader,
		int batchSize) : MyTrainer<net_type>(startingLearningRate, syncFile, outputNetworkFile, minimumLearningRate, iterationWithoutProgressTreshold, verboseMode)
	{
		this->mmodDataLoader = mmodDataLoader;
		this->chipDims = normalizedChipDims;
		this->detectorWindowTargetSize = detectorWindowTargetSize;
		this->detectorWindowMinTargetSize = detectorWindowMinTargetSize;
		this->batchSize = batchSize;
		this->cropper = new random_cropper();
		this->cropper->set_chip_dims(*this->chipDims);
		this->cropper->set_max_rotation_degrees(2);
		this->cropper->set_randomly_flip(cropperRandomlyFlip);
		this->cropper->set_min_object_size(this->detectorWindowTargetSize, this->detectorWindowMinTargetSize);
	}

	~MmodTrainer() {
		delete this->mmodDataLoader;
		delete this->cropper;
	}

private:
	chip_dims* chipDims;
	int detectorWindowTargetSize;
	int detectorWindowMinTargetSize;
	std::vector<matrix<rgb_pixel>> lastImagesToTrain;
	std::vector<std::vector<mmod_rect>> lastMmodBoxes;
	bool cycleDataset = false;
	int batchSize;

public:
	MmodDatasetLoader* mmodDataLoader;
	random_cropper* cropper;

	void obtaionNextBatchOfTrainingDataAndLabels(std::vector<input_type>& data, std::vector<training_label_type>& labels) {
		if (!MmodTrainer::mmodDataLoader->isEnd()) {
			MmodTrainer::mmodDataLoader->loadDatasetPart(lastImagesToTrain, lastMmodBoxes);
		}
		else if (cycleDataset) {
			MmodTrainer::mmodDataLoader->resetLoader();
			MmodTrainer::mmodDataLoader->loadDatasetPart(lastImagesToTrain, lastMmodBoxes);
		}

		(*cropper)(this->batchSize, lastImagesToTrain, lastMmodBoxes, data, labels);
		preprocessTrainingData(data, labels);
	}

	net_type getNetWithSpecificOptions() {
		mmod_options options(mmodDataLoader->getAllMmodRects(), this->detectorWindowTargetSize, this->detectorWindowMinTargetSize);
		
		cout << "num detector windows: " << options.detector_windows.size() << endl;
		for (auto& w : options.detector_windows)
			cout << "detector window width by height: " << w.width << " x " << w.height << endl;
		cout << "overlap NMS IOU thresh:             " << options.overlaps_nms.get_iou_thresh() << endl;
		cout << "overlap NMS percent covered thresh: " << options.overlaps_nms.get_percent_covered_thresh() << endl;
		
		net_type net(options);
		
		
		net.subnet().layer_details().set_num_filters(options.detector_windows.size());
		return net;
	}

	virtual void preprocessTrainingData(std::vector<matrix<rgb_pixel>>& imagesToTrain, std::vector<std::vector<mmod_rect>>& mmodBoxes) = 0;
};
#endif // MY_TRAINER