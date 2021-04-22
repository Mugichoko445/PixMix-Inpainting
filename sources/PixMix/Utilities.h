#pragma once

#include <opencv2/opencv.hpp>

namespace util
{
	void CreateVizPosMap(cv::InputArray srcPosMap, cv::OutputArray dstColorMap);
}