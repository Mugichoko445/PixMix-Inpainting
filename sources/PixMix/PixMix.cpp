#include "PixMix.h"

namespace dr
{
	PixMix::PixMix() { }
	PixMix::~PixMix() { }

	void PixMix::Run(cv::InputArray color, cv::InputArray mask, cv::OutputArray inpainted, const det::PixMixParams& params, bool debugViz)
	{
		assert(color.size() == mask.size());
		assert(color.type() == CV_8UC3);
		assert(mask.type() == CV_8U);

		Init(color, mask, params.blurSize);

		for (int lv = int(pm.size()) - 1; lv >= 0; --lv)
		{
			pm[lv].Run(params);
			if (lv > 0) FillInLowerLv(pm[lv], pm[lv - 1]);

#pragma region DEBUG_VIZ
			if (debugViz)
			{
				cv::Mat vizColor, vizPosMap;
				cv::resize(*pm[lv].GetColorPtr(), vizColor, color.size(), 0.0f, 0.0f, cv::INTER_NEAREST);
				util::CreateVizPosMap(*pm[lv].GetPosMapPtr(), vizPosMap);
				cv::resize(vizPosMap, vizPosMap, color.size(), 0.0f, 0.0f, cv::INTER_NEAREST);
				cv::imshow("debug - inpainted color", vizColor);
				cv::imshow("debug - colord position map", vizPosMap);
				cv::waitKey(1);
			}
#pragma endregion

		}

		BlendBorder(inpainted);
	}

	void PixMix::Init(cv::InputArray color, cv::InputArray mask, const int blurSize)
	{
		// build pyramid
		pm.resize(CalcPyrmLv(color.cols(), color.rows()));
		pm[0].Init(color.getMat(), mask.getMat());
		for (int lv = 1; lv < pm.size(); ++lv)
		{
			auto lvSize = pm[lv - 1].GetColorPtr()->size() / 2;

			// color
			cv::Mat3b tmpColor;
			cv::resize(*(pm[lv - 1].GetColorPtr()), tmpColor, lvSize, 0.0, 0.0, cv::INTER_LINEAR);
			// mask
			cv::Mat1b tmpMask;
			cv::resize(*(pm[lv - 1].GetMaskPtr()), tmpMask, lvSize, 0.0, 0.0, cv::INTER_LINEAR);
			for (int r = 0; r < tmpMask.rows; ++r)
			{
				auto ptrMask = tmpMask.ptr<uchar>(r);
				for (int c = 0; c < tmpMask.cols; ++c)
				{
					ptrMask[c] = ptrMask[c] < 255 ? 0 : 255;
				}
			}

			pm[lv].Init(tmpColor, tmpMask);
		}

		// for the final composite
		mColor = color.getMat().clone();
		cv::blur(mask, mAlpha, cv::Size(blurSize, blurSize));
	}

	int PixMix::CalcPyrmLv(int width, int height)
	{
		auto pyrmLv = 1;
		auto size = std::min(width, height);
		while ((size /= 2) >= 5) ++pyrmLv;

		return std::min(pyrmLv, 6);
	}

	void PixMix::FillInLowerLv(OneLvPixMix& pmUpper, OneLvPixMix& pmLower)
	{
		cv::Mat3b mColorUpsampled;
		cv::resize(*(pmUpper.GetColorPtr()), mColorUpsampled, pmLower.GetColorPtr()->size(), 0.0, 0.0, cv::INTER_LINEAR);
		cv::Mat2i mPosMapUpsampled;
		cv::resize(*(pmUpper.GetPosMapPtr()), mPosMapUpsampled, pmLower.GetPosMapPtr()->size(), 0.0, 0.0, cv::INTER_NEAREST);
		for (int r = 0; r < mPosMapUpsampled.rows; ++r)
		{
			auto ptr = mPosMapUpsampled.ptr<cv::Vec2i>(r);
			for (int c = 0; c < mPosMapUpsampled.cols; ++c) ptr[c] = ptr[c] * 2 + cv::Vec2i(r % 2, c % 2);
		}

		auto mColorLw = *(pmLower.GetColorPtr());
		auto mMaskLw = *(pmLower.GetMaskPtr());
		auto mPosMapLw = *(pmLower.GetPosMapPtr());

		const int wLw = pmLower.GetColorPtr()->cols;
		const int hLw = pmLower.GetColorPtr()->rows;
		for (int r = 0; r < hLw; ++r)
		{
			auto ptrColorLw = mColorLw.ptr<cv::Vec3b>(r);
			auto ptrColorUpsampled = mColorUpsampled.ptr<cv::Vec3b>(r);
			auto ptrMaskLw = mMaskLw.ptr<uchar>(r);
			auto ptrPosMapLw = mPosMapLw.ptr<cv::Vec2i>(r);
			auto ptrPosMapUpsampled = mPosMapUpsampled.ptr<cv::Vec2i>(r);
			for (int c = 0; c < wLw; ++c)
			{
				if (ptrMaskLw[c] == 0)
				{
					ptrColorLw[c] = ptrColorUpsampled[c];
					ptrPosMapLw[c] = ptrPosMapUpsampled[c];
				}
			}
		}
	}

	void PixMix::BlendBorder(cv::OutputArray dst)
	{
		cv::Mat3f mColorF, mPMColorF, mDstF(pm[0].GetColorPtr()->size());
		mColor.convertTo(mColorF, CV_32FC3, 1.0 / 255.0);
		pm[0].GetColorPtr()->convertTo(mPMColorF, CV_32FC3, 1.0 / 255.0);

		cv::Mat1f mAlphaF;
		mAlpha.convertTo(mAlphaF, CV_32F, 1.0 / 255.0);

		for (int r = 0; r < mColor.rows; ++r)
		{
			auto ptrSrc = mColorF.ptr<cv::Vec3f>(r);
			auto ptrPM = mPMColorF.ptr<cv::Vec3f>(r);
			auto ptrDst = mDstF.ptr<cv::Vec3f>(r);
			auto ptrAlpha = mAlphaF.ptr<float>(r);
			for (int c = 0; c < mColor.cols; ++c)
			{
				ptrDst[c] = ptrAlpha[c] * ptrSrc[c] + (1.0f - ptrAlpha[c]) * ptrPM[c];
			}
		}

		mDstF.convertTo(dst, CV_8UC3, 255.0);
	}
}