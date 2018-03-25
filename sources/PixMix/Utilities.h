#pragma once

#include <opencv2/opencv.hpp>

namespace Util
{
	void createVizPosMap(const cv::Mat_<cv::Vec2i> &srcPosMap, cv::Mat_<cv::Vec3b> &dstColorMap);
	void createMask(const cv::Mat_<cv::Vec3b> &srcColor, const cv::Scalar &maskColor, cv::Mat_<uchar> &dstMask, const int maskVal = 0, const int nonMaskVal = 255);
}