#pragma once

#ifndef PUPIL_TRAINER
#define PUPIL_TRAINER

#include <StdIncludes.h>
#include "MmodTrainer.h"
#include "MmodDatasetLoader.h"

template <long num_filters, typename SUBNET> using con5d = con<num_filters, 5, 5, 2, 2, SUBNET>;
template <long num_filters, typename SUBNET> using con3 = con<num_filters, 3, 3, 1, 1, SUBNET>;
template <typename SUBNET> using rcon3 = relu<bn_con<con3<32, SUBNET>>>;
template <typename SUBNET> using affineRcon3 = relu<affine<con3<32, SUBNET>>>;

template <typename SUBNET> using downsampler = relu<bn_con<con5d<32, relu<bn_con<con5d<32, relu<bn_con<con5d<32, SUBNET>>>>>>>>>;
typedef loss_mmod<con<1, 6, 6, 1, 1, rcon3<rcon3<rcon3<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>> pupil_detection_net_type;

template <typename SUBNET> using downsamplerAffine = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<32, SUBNET>>>>>>>>>;
typedef loss_mmod<con<1, 6, 6, 1, 1, affineRcon3<affineRcon3<affineRcon3<downsamplerAffine<input_rgb_image_pyramid<pyramid_down<6>>>>>>>> pupil_detection_net_type_affine;

const int MINIMUM_IMG_DIM_SIZE = 304;


class PupilTrainer : public MmodTrainer <pupil_detection_net_type>
{
public:
	PupilTrainer(double startingLearningRate,
		string syncFile,
		string outputNetworkFile,
		double minimumLearningRate,
		int iterationWithoutProgressTreshold,
		bool verboseMode,
		MmodDatasetLoader* mmodDataLoader,
		int batchSize) : MmodTrainer<pupil_detection_net_type>(startingLearningRate, syncFile, outputNetworkFile, minimumLearningRate, iterationWithoutProgressTreshold, (new chip_dims(MINIMUM_IMG_DIM_SIZE, MINIMUM_IMG_DIM_SIZE)), false, 0, verboseMode, mmodDataLoader, batchSize)
	{}

private:
	void adjustImagesSize(std::vector<matrix<rgb_pixel>>& imagesToTrain, std::vector<std::vector<mmod_rect>>& mmodBoxes) {
		std::vector<matrix<rgb_pixel>> notResized = imagesToTrain;
		std::vector<std::vector<mmod_rect>> mmodBoxesNotResized = mmodBoxes;
		
		int pyramidUpScale;
		unsigned long minimumLabelWidth;
		unsigned long minimumLabelHeight;
		unsigned long maximumLabelWidth;
		unsigned long maximumLabelHeight;

		for (int i = 0; i < imagesToTrain.size(); i++) {
			pyramidUpScale = 1;
			while (imagesToTrain[i].nr() < MINIMUM_IMG_DIM_SIZE || imagesToTrain[i].nc() < MINIMUM_IMG_DIM_SIZE) {
				pyramid_up(imagesToTrain[i]);
				pyramidUpScale *= 2;
			}
		
			for (int j = 0; j < mmodBoxes[i].size(); j++) {
				mmodBoxes[i][j].rect.set_top(mmodBoxes[i][j].rect.top() * pyramidUpScale + pyramidUpScale * 1);
				mmodBoxes[i][j].rect.set_left(mmodBoxes[i][j].rect.left() * pyramidUpScale + pyramidUpScale * 1);
				mmodBoxes[i][j].rect.set_bottom(mmodBoxes[i][j].rect.bottom() * pyramidUpScale + pyramidUpScale * 1);
				mmodBoxes[i][j].rect.set_right(mmodBoxes[i][j].rect.right() * pyramidUpScale + pyramidUpScale * 1);
			}
		}

		minimumLabelWidth = mmodBoxes[0][0].rect.width();
		minimumLabelHeight = mmodBoxes[0][0].rect.height();
		maximumLabelWidth = mmodBoxes[0][0].rect.width();
		maximumLabelHeight = mmodBoxes[0][0].rect.height();

		for (std::vector<mmod_rect> imageRects : mmodBoxes) {
			for (mmod_rect rect : imageRects) {
				minimumLabelWidth = std::min(minimumLabelWidth, rect.rect.width());
				minimumLabelHeight = std::min(minimumLabelHeight, rect.rect.height());
				maximumLabelWidth = std::max(maximumLabelWidth, rect.rect.width());
				maximumLabelHeight = std::max(maximumLabelHeight, rect.rect.height());
			}
		}

		printf("minHeight: %d \n", minimumLabelHeight);
		printf("minWidth: %d \n", minimumLabelWidth);
		printf("maxHeight: %d \n", maximumLabelHeight);
		printf("maxWidth: %d \n", maximumLabelWidth);

		this->cropper->set_min_object_size(minimumLabelWidth - 1, minimumLabelHeight - 1);
		this->detectorWindowTargetSize = minimumLabelWidth;
		this->detectorWindowMinTargetSize = minimumLabelHeight;
	}

	void preprocessTrainingData(std::vector<matrix<rgb_pixel>>& imagesToTrain, std::vector<std::vector<mmod_rect>>& mmodBoxes) {}
};
#endif // PUPIL_TRAINER

