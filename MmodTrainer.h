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
		MmodDatasetLoader* mmodDataLoader) : MyTrainer<net_type>(startingLearningRate, syncFile, outputNetworkFile, minimumLearningRate, iterationWithoutProgressTreshold, verboseMode)
	{
		this->mmodDataLoader = mmodDataLoader;
		this->chipDims = normalizedChipDims;
		this->detectorWindowTargerSize = detectorWindowTargetSize;
		this->detectorWindowMinTargetSize = detectorWindowMinTargetSize;

		cropper = new random_cropper();
		cropper->set_randomly_flip(false);
		cropper->set_chip_dims(*this->chipDims);

		cropper->set_min_object_size(this->detectorWindowTargerSize, this->detectorWindowMinTargetSize);
		cropper->set_max_rotation_degrees(0);
	}

	~MmodTrainer() {
		delete this->mmodDataLoader;
		delete this->cropper;
	}

protected:
	chip_dims* chipDims;
	int detectorWindowTargerSize;
	int detectorWindowMinTargetSize;
public:
	MmodDatasetLoader* mmodDataLoader;
	random_cropper* cropper;

	void obtaionNextBatchOfTrainingDataAndLabels(std::vector<input_type>& data, std::vector<training_label_type>& labels) {
		std::vector<matrix<rgb_pixel>> imagesToTrain;
		std::vector<std::vector<mmod_rect>> mmodBoxes;

		MmodTrainer::mmodDataLoader->loadDatasetPart(imagesToTrain, mmodBoxes);

		if (imagesToTrain.size() == 0) {
			MmodTrainer::mmodDataLoader->resetLoader();
			MmodTrainer::mmodDataLoader->loadDatasetPart(imagesToTrain, mmodBoxes);
		}

		//cropTrainingData(imagesToTrain, mmodBoxes);
		preprocessTrainingData(imagesToTrain, mmodBoxes);

		data = imagesToTrain;
		labels = mmodBoxes;
	}

	void cropTrainingData(std::vector<matrix<rgb_pixel>>& imagesToTrain, std::vector<std::vector<mmod_rect>>& mmodBoxes) {
		std::vector<matrix<rgb_pixel>> crops;
		std::vector<std::vector<mmod_rect>> crop_boxes;

		/*image_window win;
		win.clear_overlay();
		win.set_image(imagesToTrain[0]);*/

		(*cropper)(imagesToTrain.size(), imagesToTrain, mmodBoxes, crops, crop_boxes);

		/*image_window win2;
		win2.clear_overlay();
		win2.set_image(crops[0]);*/
		
		//cin.get();

		imagesToTrain = crops;
		mmodBoxes = crop_boxes;
	}

	net_type getNetWithSpecificOptions() {
		mmod_options options(mmodDataLoader->getAllMmodRects(), this->detectorWindowTargerSize, this->detectorWindowMinTargetSize);
		
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