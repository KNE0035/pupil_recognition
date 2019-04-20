#pragma once

#ifndef VIDEO_OBJECT_BOX_DETECTOR
#define VIDEO_OBJECT_BOX_DETECTOR

#include <StdIncludes.h>

class VideoObjectBoxDetector
{
public:
	VideoObjectBoxDetector();
	VideoObjectBoxDetector(string videoFilePath);

	void startDetector();
protected:
	int faceDetectionNumber = 0;
	int sampleNumber = 0;
private:
	cv::VideoCapture videoCapture;
	string videoFilePath = "";
	
	virtual std::vector<rectangle> getBoundingBoxesFromImage(cv::Mat cvimg) = 0;
};
#endif // VIDEO_OBJECT_BOX_DETECTOR

