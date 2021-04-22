#include "Utilities.h"

namespace util
{
	void CreateVizPosMap(const cv::InputArray srcPosMap, cv::OutputArray dstColorMap)
	{
		auto src = cv::Mat2i(srcPosMap.getMat());
		auto dst = cv::Mat3b(srcPosMap.size());

		for (int r = 0; r < src.rows; ++r) for (int c = 0; c < src.cols; ++c)
		{
			dst(r, c)[0] = int((float)src(r, c)[1] / (float)src.cols * 255.0f);
			dst(r, c)[1] = int((float)src(r, c)[0] / (float)src.rows * 255.0f);
			dst(r, c)[2] = 255;
		}

		dst.copyTo(dstColorMap);
	}
}