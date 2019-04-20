#include <StdIncludes.h>
#include <dlib/data_io.h>
#include "PupilTrainer.h"
#include "MmodDatasetLoader.h"
#include "VideoPupilBoxDetector.h"


using namespace std;
using namespace dlib;

void testOrTrainNetwork();

int main(int argc, char** argv) try
{
	//VideoPupilBoxDetector pupilDetector = VideoPupilBoxDetector("pupil_train_shallower_disturbed_colors.dat", "shape_predictor_68_face_landmarks.dat", "test9.mp4");
	//pupilDetector.startDetector();
	//test4, 8
	testOrTrainNetwork();
}
catch (std::exception& e)
{
	cout << e.what() << endl;
}


void testOrTrainNetwork() try {
	MmodDatasetLoader* dataLoader = new MmodDatasetLoader("C:/Users/kne0035/dev/projects/pupil_recognition/training_images", "pupil_info_square_with_borders2.xml", 3000);

	std::vector<std::vector<mmod_rect>> rects;
	rects = dataLoader->getAllMmodRects();

	//4 pyramid_up
	std::vector<matrix<rgb_pixel>> imagesToTrain;
	std::vector<std::vector<mmod_rect>> mmodBoxes;


	//PupilTrainer pupilTrainer = PupilTrainer(0.1, "pupil_train_sync", "pupil_train_shallower_disturbed_colors.dat", 1e-6, 8000, true, dataLoader);
	//pupilTrainer.train();

	pupil_detection_net_type net;
	//deserialize("pupil_train_shallower_disturbed_colors.dat") >> net;
	dataLoader->resetLoader();
	dataLoader->loadDatasetPart(imagesToTrain, mmodBoxes);
	//cout << "training results: " << test_object_detection_function(net, imagesToTrain, mmodBoxes) << endl;
	int count = 0;
	//pupilTrainer.obtaionNextBatchOfTrainingDataAndLabels(imagesToTrain, mmodBoxes);
	image_window win, win2;
	int offset = 1400;
	for (int i = offset; i < imagesToTrain.size(); ++i)
	{
		/*while (imagesToTrain[i].nr() < MINIMUM_IMG_DIM_SIZE && imagesToTrain[i].nc() < MINIMUM_IMG_DIM_SIZE) {
			pyramid_up(imagesToTrain[i]);
		}*/

		/*auto dets = net(imagesToTrain[i]);
		win.clear_overlay();
		win.set_image(imagesToTrain[i]);
		for (auto&& d : dets) {
			if (dets.size() > 1) {
				printf("asf");
			}
			win.add_overlay(d);
			count++;
		}*/

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
