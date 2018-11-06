#pragma once

#ifndef PUPIL_TRAINER
#define PUPIL_TRAINER

#include <StdIncludes.h>
#include "MmodTrainer.h"
#include "MmodDatasetLoader.h"

template <long num_filters, typename SUBNET> using con5d = con<num_filters, 5, 5, 2, 2, SUBNET>;
template <long num_filters, typename SUBNET> using con3 = con<num_filters, 3, 3, 1, 1, SUBNET>;
template <typename SUBNET> using downsampler = relu<bn_con<con5d<32, relu<bn_con<con5d<32, relu<bn_con<con5d<32, SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon3 = relu<bn_con<con3<32, SUBNET>>>;
typedef loss_mmod<con<1, 6, 6, 1, 1, rcon3<rcon3<rcon3<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>> pupil_detection_net_type;

class PupilTrainer : public MmodTrainer <pupil_detection_net_type>
{
public:
	PupilTrainer(double startingLearningRate,
		string syncFile,
		string outputNetworkFile,
		double minimumLearningRate,
		int iterationWithoutProgressTreshold,
		MmodDatasetLoader* mmodDataLoader) : MmodTrainer<pupil_detection_net_type>(startingLearningRate, syncFile, outputNetworkFile, minimumLearningRate, iterationWithoutProgressTreshold, (new chip_dims(200, 200)), 20, 20, mmodDataLoader)
	{
		cropper = new random_cropper();
		cropper->set_randomly_flip(false);
		cropper->set_chip_dims(*this->chipDims);
		cropper->set_min_object_size(this->detectorWindowTargerSize, this->detectorWindowMinTargetSize);
		cropper->set_max_rotation_degrees(0);
	}

	~PupilTrainer() {
		delete cropper;
	}

private:
	dlib::rand rnd;
	random_cropper* cropper;
	
	void preprocessTrainingData(std::vector<matrix<rgb_pixel>>& imagesToTrain, std::vector<std::vector<mmod_rect>>& mmodBoxes) {		
		std::vector<matrix<rgb_pixel>> crops;
		std::vector<std::vector<mmod_rect>> crop_boxes;
		
		(*cropper)(imagesToTrain.size(), imagesToTrain, mmodBoxes, crops, crop_boxes);
		
		for (auto&& img : crops)
			disturb_colors(img, this->rnd);

		imagesToTrain = crops;
		mmodBoxes = crop_boxes;
	}
};
#endif // PUPIL_TRAINER

