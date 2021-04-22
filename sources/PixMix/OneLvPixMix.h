#pragma once

#include <random>
#include <opencv2/opencv.hpp>

#include "Utilities.h"

namespace dr
{
	namespace det
	{
		struct PixMixParams
		{
			int maxItr = 1;				// max iteration per pyramid level
			int maxRandSearchItr = 1;	// max number of random sampling per pixel
			float alpha = 0.05f;		// balancing parameter between spatial and appearance cost
			float threshDist = 0.5f;	// 0.5 means the half of the width/height is the maximum
			int blurSize = 5;			// blur kernel size for the final composition
		};

		class OneLvPixMix
		{
		public:
			OneLvPixMix();
			~OneLvPixMix();

			void Init(const cv::Mat3b& color, const cv::Mat1b& mask);
			void Run(const PixMixParams& params);

			cv::Mat3b* GetColorPtr();
			cv::Mat1b* GetMaskPtr();
			cv::Mat2i* GetPosMapPtr();

		private:
			const int borderSize;
			const int borderSizePosMap;
			const int windowSize;

			enum { WO_BORDER = 0, W_BORDER = 1 };
			cv::Mat3b mColor[2];
			cv::Mat1b mMask[2];
			cv::Mat2i mPosMap[2];	// current position map: f

			const cv::Vec2i toLeft;
			const cv::Vec2i toRight;
			const cv::Vec2i toUp;
			const cv::Vec2i toDown;
			std::vector<cv::Vec2i> vSptAdj;

			std::mt19937 mt;
			std::uniform_int_distribution<int> cRand;
			std::uniform_int_distribution<int> rRand;

			cv::Vec2i GetValidRandPos();

			void Inpaint();

			float CalcSptCost(
				const cv::Vec2i& target,
				const cv::Vec2i& ref,
				float maxDist,		// tau_s
				float w = 0.125f	// 1.0f / 8.0f
			);
			float CalcAppCost(
				const cv::Vec2i& target,
				const cv::Vec2i& ref,
				float w = 0.04f		// 1.0f / 25.0f
			);

			void FwdUpdate(
				const float scAlpha,
				const float acAlpha,
				const float thDist,
				const int maxRandSearchItr
			);
			void BwdUpdate(
				const float scAlpha,
				const float acAlpha,
				const float thDist,
				const int maxRandSearchItr
			);
		};

		inline cv::Mat3b* OneLvPixMix::GetColorPtr()
		{
			return &(mColor[WO_BORDER]);
		}
		inline cv::Mat1b* OneLvPixMix::GetMaskPtr()
		{
			return &(mMask[WO_BORDER]);
		}
		inline cv::Mat2i* OneLvPixMix::GetPosMapPtr()
		{
			return &mPosMap[WO_BORDER];
		}

		inline cv::Vec2i OneLvPixMix::GetValidRandPos()
		{
			cv::Vec2i p;
			do {
				p = cv::Vec2i(rRand(mt), cRand(mt));
			} while (mMask[WO_BORDER](p) != 255);

			return p;
		}
	}
}