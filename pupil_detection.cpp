#include <StdIncludes.h>
#include <dlib/data_io.h>
#include "PupilTrainer.h"
#include "MmodDatasetLoader.h"
#include "VideoPupilBoxDetector.h"


using namespace std;
using namespace dlib;

void testNetwork();
void trainNetwork();

int main(int argc, char** argv) try
{
	/*VideoPupilBoxDetector pupilDetector = VideoPupilBoxDetector("pupil_train_1000_samples_conv_stride_8x_bounding_box_regression.dat", "shape_predictor_68_face_landmarks.dat", "test_video.mp4");
	pupilDetector.startDetector();*/
	trainNetwork();
	//testNetwork();
}
catch (std::exception& e)
{
	cout << e.what() << endl;
}

void trainNetwork() try {
		MmodDatasetLoader* dataLoader = new MmodDatasetLoader("C:/Users/kne0035/dev/projects/pupil_recognition/training_images", "pupil_info_training.xml", 3000);

		std::vector<matrix<rgb_pixel>> imagesToTrain;
		std::vector<std::vector<mmod_rect>> mmodBoxes;


		PupilTrainer pupilTrainer = PupilTrainer(0.1, "pupil_train_sync", "pupil_train_output.dat", 1e-6, 8000, true, dataLoader, 32);
		pupilTrainer.train();

		pupil_detection_net_type net;
		deserialize("test.dat") >> net;
		dataLoader->resetLoader();
		dataLoader->loadDatasetPart(imagesToTrain, mmodBoxes);

		int pyramidUpScale;
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

		cout << "training results: " << test_object_detection_function(net, imagesToTrain, mmodBoxes) << endl;

		int count = 0;
		image_window win, win2;
		int offset = 0;
		float avgIOU = 0;
		for (int i = offset; i < imagesToTrain.size(); ++i)
		{
			while (imagesToTrain[i].nr() < MINIMUM_IMG_DIM_SIZE || imagesToTrain[i].nc() < MINIMUM_IMG_DIM_SIZE) {
				pyramid_up(imagesToTrain[i]);
			}

			auto dets = net(imagesToTrain[i]);
			win.clear_overlay();
			win.set_image(imagesToTrain[i]);
			for (auto&& d : dets) {
				win.add_overlay(d);
				count++;

				float iou = d.rect.intersect(mmodBoxes[i][0].rect).area() / (float)(d.rect + mmodBoxes[i][0].rect).area();
				avgIOU += iou;
			}

			cout << i + 1 << ". obrazek" << endl;
			cin.get();

			cout << "Hit enter to process the next image." << endl;
		}

		cout << endl << count << "/" << imagesToTrain.size();

		printf("\n absolute error %f", avgIOU);
		avgIOU /= count;
		printf("\nerror %f", avgIOU);
	}
	catch (std::exception& e)
	{
		cout << e.what() << endl;

	}

void testNetwork() try {
	MmodDatasetLoader* dataLoader = new MmodDatasetLoader("C:/Users/kne0035/dev/projects/pupil_recognition/training_images", "pupil_info_testing.xml", 3000);

	std::vector<matrix<rgb_pixel>> imagesToTrain;
	std::vector<std::vector<mmod_rect>> mmodBoxes;

	pupil_detection_net_type net;
	deserialize("pupil_train_1000_samples_conv_stride_8x_bounding_box_regression.dat") >> net;
	dataLoader->loadDatasetPart(imagesToTrain, mmodBoxes);

	int pyramidUpScale;
	for (int i = 0; i < imagesToTrain.size(); i++) {
		pyramidUpScale = 1;
		while (imagesToTrain[i].nr() < MINIMUM_IMG_DIM_SIZE || imagesToTrain[i].nc() < MINIMUM_IMG_DIM_SIZE) {
			pyramid_up(imagesToTrain[i]);
			pyramidUpScale *= 2;
		}
	}

	int count = 0;
	image_window win;
	int offset = 0;
	for (int i = offset; i < imagesToTrain.size(); ++i)
	{
		while (imagesToTrain[i].nr() < MINIMUM_IMG_DIM_SIZE || imagesToTrain[i].nc() < MINIMUM_IMG_DIM_SIZE) {
			pyramid_up(imagesToTrain[i]);
		}

		auto dets = net(imagesToTrain[i]);
		win.clear_overlay();
		win.set_image(imagesToTrain[i]);
		for (auto&& d : dets) {
			win.add_overlay(d);
			count++;
		}

		cout << i + 1 << ". obrazek" << endl;
		//cin.get();

		cout << "Hit enter to process the next image." << endl;
	}

	cout << endl << count << "/" << imagesToTrain.size();
}
catch (std::exception& e)
{
	cout << e.what() << endl;

}