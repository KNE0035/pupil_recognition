#include <StdIncludes.h>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include "PupilTrainer.h"
#include "MmodDatasetLoader.h"

using namespace std;
using namespace dlib;

int main(int argc, char** argv) try
{
	MmodDatasetLoader* dataLoader = new MmodDatasetLoader("C:/Users/kne0035/dev/projects/pupil_recognition/training_images", "pupil_info.xml", 125);

	PupilTrainer pupilTrainer = PupilTrainer(0.1, "pupil_train_sync", "mmod_pupil_network.dat", 1e-4, 300, true, dataLoader);
	pupilTrainer.train();
}
catch (std::exception& e)
{
	cout << e.what() << endl;
}

