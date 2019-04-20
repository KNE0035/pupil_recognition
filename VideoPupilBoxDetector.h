#pragma once

#ifndef VIDEO_PUPIL_BOX_DETECTOR
#define VIDEO_PUPIL_BOX_DETECTOR

#include <StdIncludes.h>
#include "VideoObjectBoxDetector.h"
#include "PupilTrainer.h"
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>

class VideoPupilBoxDetector : public VideoObjectBoxDetector
{
public:
	VideoPupilBoxDetector(string pupilNetDatFilePath, string faceShapePredictorDatFilePath);
	VideoPupilBoxDetector(string pupilNetDatFilePath, string  faceShapePredictorDatFilePath, string videoFilePath);
protected:

private:
	
	
	pupil_detection_net_type_affine pupilNet;
	frontal_face_detector frontalFaceDetector = dlib::get_frontal_face_detector();
	shape_predictor faceShapePredictor;

	std::vector<rectangle> getBoundingBoxesFromImage(cv::Mat cvimg);
	rectangle getPupilBoxFromEyeArea(cv::Mat cvimg, rectangle eyeArea);
};
#endif // VIDEO_PUPIL_BOX_DETECTOR

