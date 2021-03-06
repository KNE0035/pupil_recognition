#include "StdIncludes.h"
#include "VideoPupilBoxDetector.h"
#include <dlib/opencv.h>

VideoPupilBoxDetector::VideoPupilBoxDetector(string pupilNetDatFilePath, string faceShapePredictorDatFilePath) {
	deserialize(pupilNetDatFilePath) >> pupilNet;
	deserialize(faceShapePredictorDatFilePath) >> faceShapePredictor;
}

VideoPupilBoxDetector::VideoPupilBoxDetector(string pupilNetDatFilePath, string  faceShapePredictorDatFilePath, string videoFilePath) : VideoObjectBoxDetector(videoFilePath) {
	deserialize(pupilNetDatFilePath) >> pupilNet;
	deserialize(faceShapePredictorDatFilePath) >> faceShapePredictor;
}

std::vector<rectangle> VideoPupilBoxDetector::getBoundingBoxesFromImage(cv::Mat cvimg) {
	std::vector<rectangle> pupilBoxes;
	cv_image <bgr_pixel> dlibStructImg(cvimg);
	std::vector<rectangle> faces = this->frontalFaceDetector(dlibStructImg);

	std::vector<full_object_detection> shapes;
	for (unsigned long i = 0; i < faces.size(); ++i) {
		shapes.push_back(faceShapePredictor(dlibStructImg, faces[i]));
	}
		
	
	int xOffset = 0;
	int yOffsetUp = cvimg.rows * 0.03;
	int yOffsetDown = cvimg.rows * 0.02;

	for (full_object_detection shape : shapes) {
		faceDetectionNumber++;
		rectangle leftEyeArea = rectangle(shape.part(36).x() - xOffset, 
									     (shape.part(36).y() + shape.part(39).y()) * 0.5 - yOffsetUp, 
									     shape.part(39).x() + xOffset, 
									     (shape.part(36).y() + shape.part(39).y()) * 0.5 + yOffsetDown);


		rectangle rightEyeArea = rectangle(shape.part(42).x() - xOffset,
										  (shape.part(42).y() + shape.part(45).y()) * 0.5 - yOffsetUp,
										  shape.part(45).x() + xOffset,
										  (shape.part(42).y() + shape.part(45).y()) * 0.5 + yOffsetDown);

		rectangle leftPupilBox = getPupilBoxFromEyeArea(cvimg ,leftEyeArea);
		rectangle rightPupilBox = getPupilBoxFromEyeArea(cvimg, rightEyeArea);

		if (leftPupilBox.right() > 0) {
			pupilBoxes.push_back(leftPupilBox);
		}

		if (rightPupilBox.right() > 0) {
			pupilBoxes.push_back(rightPupilBox);
		}
	}
	
	return pupilBoxes;
}

rectangle VideoPupilBoxDetector::getPupilBoxFromEyeArea(cv::Mat cvimg, rectangle eyeArea) {
	cv::Mat leftEyeImg = cvimg(cv::Range(eyeArea.top(), eyeArea.bottom()), cv::Range(eyeArea.left(), eyeArea.right()));
	cv_image<bgr_pixel> cvLeft(leftEyeImg);

	matrix<unsigned char> mat;
	assign_image(mat, cvLeft);
	
	while (mat.nr() < MINIMUM_IMG_DIM_SIZE && mat.nc() < MINIMUM_IMG_DIM_SIZE) {
		pyramid_up(mat);
	}
	double enlargeRatio = 1 / round(mat.nr() / (double)eyeArea.height());

	matrix<rgb_pixel> coloredMat;
	assign_image(coloredMat, mat);

	auto dets = pupilNet(coloredMat);

	rectangle searchedRect;

	if (dets.size() > 0) {
		searchedRect = dets[0];
	}

	rectangle resultRect;
	if (searchedRect.right() > 0) {
			resultRect = rectangle(eyeArea.left() + searchedRect.left() * enlargeRatio,
								   eyeArea.top() + searchedRect.top() * enlargeRatio,
								   eyeArea.left() + searchedRect.right() * enlargeRatio,
								   eyeArea.top() + searchedRect.bottom() * enlargeRatio);
	}

	return resultRect;
}