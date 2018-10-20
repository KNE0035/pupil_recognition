#include "utils.h"
#include "ImageLoader.h"
#include <dlib/data_io.h>
#include <windows.h>
#include <tchar.h> 
#include <stdio.h>
#include <strsafe.h>

ImageLoader::ImageLoader(string imageDatasetDirectory, std::vector<string> directoryParts, int bufferSize) {
	for (const auto& directoryPart : directoryParts)
		this->directoryParts.push(directoryPart);

	this->imageDatasetDirectory = imageDatasetDirectory;
	this->bufferSize = bufferSize;

	this->provideSubdirectoryImageNamesOfNextPart();
}

ImageLoader::~ImageLoader() {

}

std::vector<PupilImageInfo> ImageLoader::LoadImagePart() {
	std::vector<string> directoryFileNames;
	std::vector<PupilImageInfo> pupilImagesInfo;

	for (int i = 0; i < this->bufferSize; i++) {
		matrix<rgb_pixel> image;
		PupilImageInfo pupilImageInfo;

		load_image(image, this->imageDatasetDirectory + "/" + this->currentSubDirName +  "/" + this->subdirectoryImageNames.front());
		

		pupilImageInfo.image = image;
		pupilImageInfo.imageName = this->subdirectoryImageNames.front();
		pupilImageInfo.setPupilInfoFromImageName(this->subdirectoryImageNames.front());

		pupilImagesInfo.push_back(pupilImageInfo);
		this->subdirectoryImageNames.pop();

		if (this->subdirectoryImageNames.empty()) {
			this->provideSubdirectoryImageNamesOfNextPart();
			return pupilImagesInfo;
		}
	}
	
	return pupilImagesInfo;
}


void ImageLoader::provideSubdirectoryImageNamesOfNextPart() {
	clearQueue(this->subdirectoryImageNames);
	this->currentSubDirName = this->directoryParts.front();
	string subdirectoryImagesPath = this->imageDatasetDirectory + "/" + this->currentSubDirName;

	this->directoryParts.pop();
	for (const auto& name : getFileNamesOfDir(subdirectoryImagesPath))
		this->subdirectoryImageNames.push(name);
}

std::vector<string> ImageLoader::getFileNamesOfDir(const string dir)
{
	std::vector<string> file_names;
	WIN32_FIND_DATA ffd;
	LARGE_INTEGER filesize;
	TCHAR szDir[MAX_PATH];
	size_t length_of_arg;
	HANDLE hFind = INVALID_HANDLE_VALUE;
	DWORD dwError = 0;

	StringCchLength(dir.c_str(), MAX_PATH, &length_of_arg);

	StringCchCopy(szDir, MAX_PATH, dir.c_str());
	StringCchCat(szDir, MAX_PATH, TEXT("\\*"));

	hFind = FindFirstFile(szDir, &ffd);

	if (INVALID_HANDLE_VALUE == hFind)
	{
		printf("error");
	}

	do
	{
		if (!(ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
		{
			file_names.push_back(ffd.cFileName);
		}
	}
	while (FindNextFile(hFind, &ffd) != 0);

	dwError = GetLastError();
	return file_names;
}