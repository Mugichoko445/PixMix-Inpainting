#pragma once

#include <vector>
#include "OneLvPixMix.h"

namespace dr
{
	using namespace det;

	class PixMix
	{
	public:
		PixMix();
		~PixMix();

		void Run(cv::InputArray color, cv::InputArray mask, cv::OutputArray inpainted, const PixMixParams& params, bool debugViz = false);

	private:
		std::vector<OneLvPixMix> pm;
		cv::Mat3b mColor;
		cv::Mat1b mAlpha;

		void Init(cv::InputArray color, cv::InputArray mask, const int blurSize);
		int CalcPyrmLv(int width, int height);
		void FillInLowerLv(OneLvPixMix& pmUpper, OneLvPixMix& pmLower);
		void BlendBorder(cv::OutputArray dst);
	};
}