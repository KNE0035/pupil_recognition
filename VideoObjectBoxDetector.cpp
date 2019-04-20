#include "VideoObjectBoxDetector.h"
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <opencv2/videoio.hpp>

VideoObjectBoxDetector::VideoObjectBoxDetector() {
	this->videoCapture = cv::VideoCapture(0);
}

VideoObjectBoxDetector::VideoObjectBoxDetector(string videoFilePath) {
	this->videoFilePath = videoFilePath;

	if (videoFilePath == "") {
		this->videoCapture = cv::VideoCapture(0);
	}
	else {
		this->videoCapture = cv::VideoCapture(videoFilePath);
	}
}

void VideoObjectBoxDetector::startDetector() {
	try
	{
		if (!videoCapture.isOpened())
		{
			if (videoFilePath == "") {
				cerr << "Unable to connect to camera" << endl;
			}
			else {
				cerr << "Unable to open file: " << this->videoFilePath << endl;
			}
			return;
		}

		image_window win;

		while(!win.is_closed())
			{
				cv::Mat cvimg;
				cv::Mat cvimgResized;
				if (!videoCapture.read(cvimg))
				{
					break;
				}
				//cv::rotate(cvimg, cvimg, 1);
				//resize(cvimg, cvimg, cv::Size(960, 540), 0, 0, cv::INTER_LINEAR_EXACT);

				cv_image<bgr_pixel> dlibStructImg(cvimg);

				std::vector<rectangle> objectsBoundingBoxes = getBoundingBoxesFromImage(cvimg);
				win.clear_overlay();
				win.set_image(dlibStructImg);

				for (unsigned long i = 0; i < objectsBoundingBoxes.size(); ++i) {
					win.add_overlay(objectsBoundingBoxes[i]);
					//cin.get();
				}
			}
    }
    catch(exception& e)
    {
        cout << e.what() << endl;
    }
}