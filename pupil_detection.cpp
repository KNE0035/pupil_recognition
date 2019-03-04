#include <StdIncludes.h>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include "PupilTrainer.h"
#include "MmodDatasetLoader.h"

using namespace std;
using namespace dlib;

int main(int argc, char** argv) try
{
	MmodDatasetLoader* dataLoader = new MmodDatasetLoader("C:/Users/kne0035/dev/projects/pupil_recognition/training_images", "pupil_info_square_with_borders.xml", 1000);

	std::vector<std::vector<mmod_rect>> rects;
	rects = dataLoader->getAllMmodRects();
	int minRectWidth = 500;
	int minRectHeight = 500;
	int imgWidth = 1000;
	int minImgHeight = 1000;

	int maxHe = 0;
	int maxwi = 0;

	std::vector<matrix<rgb_pixel>> imagesToTrain;
	std::vector<std::vector<mmod_rect>> mmodBoxes;

	for (std::vector<mmod_rect> rects1 : rects) {
		for (mmod_rect rect : rects1) {
			int width = rect.rect.width();
			int height = rect.rect.height();

			if (minRectWidth > width) {
				minRectWidth = width;
			}

			if (minRectHeight > height) {
				minRectHeight = height;
			}

			if (maxHe < height) {
				maxHe = height;
			}

			if (maxwi < width) {
				maxwi = width;
			}
		}
	}

	printf("minheight: %d \n", minRectHeight);
	printf("minWidth: %d \n", minRectWidth);
	printf("maxheight: %d \n", maxHe);
	printf("maxWidth: %d \n", maxwi);

	PupilTrainer pupilTrainer = PupilTrainer(0.1, "pupil_train_sync", "pupil_network_downsampler_off_8000_cycle.dat", 1e-6, 8000, true, dataLoader);
	pupilTrainer.train();

	pupil_detection_net_type net;
	deserialize("pupil_network_downsampler_off_8000_cycle.dat") >> net;
	dataLoader->loadDatasetPart(imagesToTrain, mmodBoxes);

	cout << "training results: " << test_object_detection_function(net, imagesToTrain, mmodBoxes) << endl;
	int count = 0;
	
	image_window win, win2;
	int offset = 0;

	for (int i = offset; i < imagesToTrain.size(); ++i)
	{
		//pyramid_up(imagesToTrain[i]);
		//pyramid_up(oneImage);
		//pyramid_up(oneImage);
		
		auto dets = net(imagesToTrain[i]);
		win.clear_overlay();
		win.set_image(imagesToTrain[i]);
		for (auto&& d : dets) {
			win.add_overlay(d);
			count++;
		}

		win2.clear_overlay();
		win2.set_image(imagesToTrain[i]);
		win2.add_overlay(mmodBoxes[i][0].rect);
		

		cout << i + 1 << ". obrazek" << endl;
		cin.get();

		cout << "Hit enter to process the next image." << endl;
	}
	
	cout << endl << count;
}
catch (std::exception& e)
{
	cout << e.what() << endl;
}
