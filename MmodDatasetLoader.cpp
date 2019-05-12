#include "MmodDatasetLoader.h"
#include <dlib/data_io.h>
#include <stdio.h>

MmodDatasetLoader::MmodDatasetLoader(string imageDatasetDirectory, string metadataDatasetFile, int bufferSize) {
	this->imageDatasetDirectory = imageDatasetDirectory;
	this->bufferSize = bufferSize;

	dlib::image_dataset_metadata::load_image_dataset_metadata(this->metadata, imageDatasetDirectory + "/" + metadataDatasetFile);
}

MmodDatasetLoader::~MmodDatasetLoader() {

}

bool MmodDatasetLoader::isEnd() {
	return lastIndexInDataset >= metadata.images.size();
}

void MmodDatasetLoader::loadDatasetPart(std::vector<matrix<rgb_pixel>>& images, std::vector<std::vector<mmod_rect> >& object_locations) {
	matrix<rgb_pixel> img;
	std::vector<mmod_rect> rects;
	images.clear();
	object_locations.clear();

	int i = lastIndexInDataset;
	for (int bufferIndex = 0; i < metadata.images.size() && bufferIndex < this->bufferSize; ++i, ++bufferIndex)
	{
		rects.clear();
		for (unsigned long j = 0; j < metadata.images[i].boxes.size(); ++j)
		{
			rects.push_back(mmod_rect(metadata.images[i].boxes[j].rect));
			rects.back().label = metadata.images[i].boxes[j].label;
		}

		load_image(img, this->imageDatasetDirectory + "/" + metadata.images[i].filename);
		images.push_back(std::move(img));
		object_locations.push_back(std::move(rects));
	}
	lastIndexInDataset = i;
}

std::vector<std::vector<mmod_rect>> MmodDatasetLoader::getAllMmodRects() {
	std::vector<std::vector<mmod_rect> > object_locations;
	std::vector<mmod_rect> rects;

	for (int i = 0; i < metadata.images.size(); i++)
	{
		for (unsigned long j = 0; j < metadata.images[i].boxes.size(); ++j)
		{
			rects.push_back(mmod_rect(metadata.images[i].boxes[j].rect));
			rects.back().label = metadata.images[i].boxes[j].label;
		}
		object_locations.push_back(std::move(rects));
	}
	return object_locations;
}

void MmodDatasetLoader::resetLoader() {
	this->lastIndexInDataset = 0;
}