#pragma once

#include <vector>
#include "OneLvPixMix.h"

class PixMix
{
public:
	PixMix();
	~PixMix();

	void init(const cv::Mat_<cv::Vec3b> &color, const cv::Mat_<uchar> &mask);
	void execute(cv::Mat_<cv::Vec3b> &dst, const float alpha);

private:
	std::vector<OneLvPixMix> pm;
	cv::Mat_<cv::Vec3b> mColor;
	cv::Mat_<uchar> mAlpha;

	int calcPyrmLv(int width, int height);
	void fillInLowerLv(OneLvPixMix &pmUpper, OneLvPixMix &pmLower);
	void blendBorder(cv::Mat_<cv::Vec3b> &dst);
};