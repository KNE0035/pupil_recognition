#include <StdIncludes.h>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include "PupilTrainer.h"
#include "MmodDatasetLoader.h"

using namespace std;
using namespace dlib;

int main(int argc, char** argv) try
{
	MmodDatasetLoader* dataLoader = new MmodDatasetLoader("C:/Users/kne0035/dev/projects/pupil_recognition/training_images", "pupil_info_square_with_borders.xml", 10000);

	std::vector<std::vector<mmod_rect>> rects;
	rects = dataLoader->getAllMmodRects();
	int minRectWidth = 500;
	int minRectHeight = 500;
	int imgWidth = 1000;
	int minImgHeight = 1000;

	int maxHe = 0;
	int maxwi = 0;

	std::vector<matrix<rgb_pixel>> imagesToTrainsss;
	std::vector<std::vector<mmod_rect>> mmodBoxesss;

	/*dataLoader->loadDatasetPart(imagesToTrainsss, mmodBoxesss);

	for (matrix<rgb_pixel> img : imagesToTrainsss) {
		cout << img.nr() << " * " << img.nc() << endl;
	}*/

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

	PupilTrainer pupilTrainer = PupilTrainer(0.1, "pupil_train_sync", "mmod_pupil_network_8000_iter_new_net_cropper_batch.dat", 1e-6, 8000, true, dataLoader);
	//23 zatim nejlepsi
	//pupilTrainer.train();

	pupil_detection_net_type net;
	deserialize("mmod_pupil_network_8000_iter_new_net_cropper_batch.dat") >> net;

	pupilTrainer.obtaionNextBatchOfTrainingDataAndLabels(imagesToTrainsss, mmodBoxesss);

	test_box_overlap overlap_tester(0.5);
	cout << "training results: " << test_object_detection_function(net, imagesToTrainsss, mmodBoxesss, overlap_tester) << endl;

	std::vector<matrix<rgb_pixel>> imagesToTrain;
	std::vector<std::vector<mmod_rect>> mmodBoxes;


	MmodDatasetLoader* dataLoader2 = new MmodDatasetLoader("C:/Users/kne0035/dev/projects/pupil_recognition/training_images", "pupil_info_square_with_borders.xml", 1300);
	dataLoader2->loadDatasetPart(imagesToTrain, mmodBoxes);
	int count = 0;

	
	pupilTrainer.obtaionNextBatchOfTrainingDataAndLabels(imagesToTrainsss, mmodBoxesss);

	matrix<rgb_pixel> img;

	image_window win, win2;

	int offset = 83;

	for (int i = offset; i < imagesToTrain.size(); ++i)
	{
		std::vector<matrix<rgb_pixel>> imagesToTrain1;
		std::vector<std::vector<mmod_rect>> mmodBoxes1;

		std::vector<matrix<rgb_pixel>> imagesToTrain2;
		std::vector<std::vector<mmod_rect>> mmodBoxes2;

		imagesToTrain1.push_back(imagesToTrain[i]);
		mmodBoxes1.push_back(mmodBoxes[i]);

		matrix<rgb_pixel> oneImage = imagesToTrain[i];
		(*(pupilTrainer.cropper))(1, imagesToTrain1, mmodBoxes1, imagesToTrain2, mmodBoxes2);

		pyramid_up(oneImage);
		//pyramid_up(oneImage);
		//pyramid_up(oneImage);
		auto dets = net(imagesToTrain2[0]);
		win.clear_overlay();
		win2.clear_overlay();
		win2.set_image(imagesToTrain2[0]);

		win2.add_overlay(mmodBoxes2[0][0].rect);
		win.set_image(imagesToTrain2[0]);
		
		/*for (auto&& d : dets) {
			win.add_overlay(d);
			count++;
		}*/
		if (dets.size() != 0) {
			//cin.get();
		}

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
