#pragma once

#ifndef IMAGE_LOADER
#define IMAGE_LOADER

#include <StdIncludes.h>

class MmodDatasetLoader
{
private:
	int bufferSize;
	
	int lastIndexInDataset = 0;
	dlib::image_dataset_metadata::dataset metadata;
	string imageDatasetDirectory;
public:
	MmodDatasetLoader(string imageDatasetDirectory, string metadataDatasetFile, int bufferSize);

	void LoadDatasetPart(std::vector<matrix<rgb_pixel>>& images, std::vector<std::vector<mmod_rect> >& object_locations);

	void resetLoader();
	~MmodDatasetLoader();
};
#endif // IMAGE_LOADER

