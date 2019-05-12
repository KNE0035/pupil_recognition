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

	void loadDatasetPart(std::vector<matrix<rgb_pixel>>& images, std::vector<std::vector<mmod_rect>>& object_locations);
	std::vector<std::vector<mmod_rect>> getAllMmodRects();
	
	bool isEnd();

	void resetLoader();
	~MmodDatasetLoader();
};
#endif // IMAGE_LOADER

