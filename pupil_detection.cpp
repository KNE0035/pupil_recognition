#include <StdIncludes.h>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include "PupilTrainer.h"
#include "MmodDatasetLoader.h"

using namespace std;
using namespace dlib;

int main(int argc, char** argv) try
{
	MmodDatasetLoader* dataLoader = new MmodDatasetLoader("C:/Users/kne0035/dev/projects/pupil_recognition/training_images", "pupil_info.xml", 1000);
	
	std::vector<std::vector<mmod_rect>> rects;
	rects = dataLoader->getAllMmodRects();
	int minRectWidth = 500;
	int minRectHeight = 500;

	int maxHe = 0; 
	int maxwi = 0;

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

	PupilTrainer pupilTrainer = PupilTrainer(0.1, "pupil_train_sync", "mmod_pupil_network.dat", 1e-6, 1000, true, dataLoader);
	//pupilTrainer.train();

	pupil_detection_net_type net;
	deserialize("mmod_pupil_network.dat") >> net;
	
	std::vector<matrix<rgb_pixel>> imagesToTrainsss;
	std::vector<std::vector<mmod_rect>> mmodBoxesss;

	pupilTrainer.obtaionNextBatchOfTrainingDataAndLabels(imagesToTrainsss, mmodBoxesss);

	test_box_overlap overlap_tester(0.4);
	cout << "training results: " << test_object_detection_function(net, imagesToTrainsss, mmodBoxesss, overlap_tester) << endl;

	std::vector<matrix<rgb_pixel>> imagesToTrain;
	std::vector<std::vector<mmod_rect>> mmodBoxes;


	MmodDatasetLoader* dataLoader2 = new MmodDatasetLoader("C:/Users/kne0035/dev/projects/pupil_recognition/training_images", "pupil_info2.xml", 80);
	dataLoader2->loadDatasetPart(imagesToTrain, mmodBoxes);

	pupilTrainer.obtaionNextBatchOfTrainingDataAndLabels(imagesToTrainsss, mmodBoxesss);
	
	matrix<rgb_pixel> img;

	image_window win, win2;
	for (int i = 0; i < imagesToTrain.size(); ++i)
	{
		matrix<rgb_pixel> oneImage = imagesToTrain[i];
		
		auto dets = net(oneImage);
		win.clear_overlay();
		win2.clear_overlay();
		win2.set_image(oneImage);
		win.set_image(oneImage);
		for (auto&& d : dets)
			win.add_overlay(d);

		cout << "Hit enter to process the next image." << endl;
		cin.get();
	}
}
catch (std::exception& e)
{
	cout << e.what() << endl;
}

