#include "Utilities.h"

void Util::createVizPosMap(
	const cv::Mat_<cv::Vec2i> &srcPosMap,
	cv::Mat_<cv::Vec3b> &dstColorMap
)
{
	dstColorMap = cv::Mat_<cv::Vec3b>(srcPosMap.size());

	for (int r = 0; r < srcPosMap.rows; ++r)
	{
		for (int c = 0; c < srcPosMap.cols; ++c)
		{
			dstColorMap(r, c)[0] = int((float)srcPosMap(r, c)[1] / (float)srcPosMap.cols * 255.0f);
			dstColorMap(r, c)[1] = int((float)srcPosMap(r, c)[0] / (float)srcPosMap.rows * 255.0f);
			dstColorMap(r, c)[2] = 255;
		}
	}
}

void Util::createMask(
	const cv::Mat_<cv::Vec3b> &srcColor,
	const cv::Scalar &maskColor,
	cv::Mat_<uchar> &dstMask,
	const int maskVal,
	const int nonMaskVal
)
{
	dstMask = cv::Mat_<uchar>(srcColor.size());

	for (int r = 0; r < srcColor.rows; ++r)
	{
		for (int c = 0; c < srcColor.cols; ++c)
		{
			cv::Vec3b color = srcColor(r, c);

			if (color[0] == maskColor[0] && color[1] == maskColor[1] && color[2] == maskColor[2])
			{
				dstMask(r, c) = maskVal;
			}
			else
			{
				dstMask(r, c) = nonMaskVal;
			}
		}
	}
};