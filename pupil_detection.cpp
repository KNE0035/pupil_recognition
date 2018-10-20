// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
	This is an example illustrating the use of the deep learning tools from the
	dlib C++ Library.  In it, we will train the venerable LeNet convolutional
	neural network to recognize hand written digits.  The network will take as
	input a small image and classify it as one of the 10 numeric digits between
	0 and 9.

	The specific network we will run is from the paper
		LeCun, Yann, et al. "Gradient-based learning applied to document recognition."
		Proceedings of the IEEE 86.11 (1998): 2278-2324.
	except that we replace the sigmoid non-linearities with rectified linear units.

	These tools will use CUDA and cuDNN to drastically accelerate network
	training and testing.  CMake should automatically find them if they are
	installed and configure things appropriately.  If not, the program will
	still run but will be much slower to execute.
*/

#include <StdIncludes.h>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include "ImageLoader.h"


using namespace std;
using namespace dlib;

int main(int argc, char** argv) try
{
	std::vector<string>* subdirectory_parts = new std::vector<string>();
	std::vector<PupilImageInfo> results, result2;
	subdirectory_parts->push_back("s0001");

	ImageLoader* dataLoader = new ImageLoader("C:/Users/kne0035/dev/projects/pupil_recognition/training_images", *subdirectory_parts, 250);
	results = dataLoader->LoadImagePart();
	result2 = dataLoader->LoadImagePart();
	delete dataLoader;
	delete subdirectory_parts;

	subdirectory_parts = NULL;
	dataLoader = NULL;
}
catch (std::exception& e)
{
	cout << e.what() << endl;
}

