#pragma once

#ifndef IMAGE_LOADER
#define IMAGE_LOADER

#include <StdIncludes.h>
#include <PupilImageInfo.h>

class ImageLoader
{
private:
	int bufferSize;
	
	string imageDatasetDirectory;
	string currentSubDirName;
	std::queue<string> directoryParts;
	std::queue<string> subdirectoryImageNames;


public:
	ImageLoader(string imageDatasetDirectory, std::vector<string> directoryParts, int bufferSize);

	std::vector<PupilImageInfo> LoadImagePart();
	~ImageLoader();

	void provideSubdirectoryImageNamesOfNextPart();
	std::vector<string> getFileNamesOfDir(const string dir);


};
#endif // IMAGE_LOADER

