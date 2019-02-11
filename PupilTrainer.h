#pragma once

#ifndef PUPIL_TRAINER
#define PUPIL_TRAINER

#include <StdIncludes.h>
#include "MmodTrainer.h"
#include "MmodDatasetLoader.h"

/*template <long num_filters, typename SUBNET> using con5d = con<num_filters, 5, 5, 2, 2, SUBNET>;
template <long num_filters, typename SUBNET> using con5 = con<num_filters, 5, 5, 1, 1, SUBNET>;
template <long num_filters, typename SUBNET> using con3 = con<num_filters, 3, 3, 1, 1, SUBNET>;
template <typename SUBNET> using rcon3 = relu<bn_con<con3<32, SUBNET>>>;
template <typename SUBNET> using downsampler = relu<bn_con<con5d<32, relu<bn_con<con5d<32, relu<bn_con<con5d<32, SUBNET>>>>>>>>>;
typedef loss_mmod<con<1, 6, 6, 1, 1, rcon3<rcon3<rcon3<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>> pupil_detection_net_type;*/



typedef loss_mmod<
	con < 1, 6, 6, 1, 1,
	max_pool<2, 2, 2, 2, relu<con<16, 5, 5, 1, 1,
	max_pool<2, 2, 2, 2, relu<con<6, 5, 5, 1, 1,
	input_rgb_image_pyramid<pyramid_down<6>>
	>>>>>>>> pupil_detection_net_type2;

template <long num_filters, typename SUBNET> using con5d = con<num_filters, 5, 5, 2, 2, SUBNET>;
template <long num_filters, typename SUBNET> using con5 = con<num_filters, 5, 5, 1, 1, SUBNET>;
template <typename SUBNET> using downsampler = relu<bn_con<con5d<32, relu<bn_con<con5d<32, relu<bn_con<con5d<16, SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5 = relu<bn_con<con5<55, SUBNET>>>;
typedef loss_mmod<con<1, 9, 9, 1, 1, rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>> pupil_detection_net_type;


class PupilTrainer : public MmodTrainer <pupil_detection_net_type>
{
public:
	PupilTrainer(double startingLearningRate,
		string syncFile,
		string outputNetworkFile,
		double minimumLearningRate,
		int iterationWithoutProgressTreshold,
		bool verboseMode,
		MmodDatasetLoader* mmodDataLoader) : MmodTrainer<pupil_detection_net_type>(startingLearningRate, syncFile, outputNetworkFile, minimumLearningRate, iterationWithoutProgressTreshold, (new chip_dims(300, 300)), 34, 34, false, 0, verboseMode, mmodDataLoader, 100)
		//: MmodTrainer<pupil_detection_net_type>(startingLearningRate, syncFile, outputNetworkFile, minimumLearningRate, iterationWithoutProgressTreshold, (new chip_dims(155, 155)), 14, 14, false, 0, verboseMode, mmodDataLoader)
	{}

private:
	dlib::rand rnd;
	
	void preprocessTrainingData(std::vector<matrix<rgb_pixel>>& imagesToTrain, std::vector<std::vector<mmod_rect>>& mmodBoxes) {
		for (auto&& img : imagesToTrain)
			disturb_colors(img, this->rnd);
	}
};
#endif // PUPIL_TRAINER

