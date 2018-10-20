#include "utils.h"

void clearQueue(std::queue<string> &q) {
	std::queue<string> empty;
	std::swap(q, empty);
}