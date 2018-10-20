#pragma once

#ifndef PUPILS_IMAGE_INFO
#define PUPILS_IMAGE_INFO

#include <StdIncludes.h>

struct  PupilImageInfo {
	matrix<rgb_pixel> image;
	string imageName;
	bool is_pupil_visible;

	void setPupilInfoFromImageName(string imageName) {

	}
};

#endif