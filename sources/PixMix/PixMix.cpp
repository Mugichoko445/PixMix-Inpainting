#include "PixMix.h"

PixMix::PixMix() { }
PixMix::~PixMix() { }

void PixMix::init(
	const cv::Mat_<cv::Vec3b> &color,
	const cv::Mat_<uchar> &mask,
	const int blurSize
)
{
	assert(color.size() == mask.size());

	pm.resize(calcPyrmLv(color.cols, color.rows));

	pm[0].init(color, mask);
	for (int lv = 1; lv < pm.size(); ++lv)
	{
		cv::Size lvSize = pm[lv - 1].getColorPtr()->size() / 2;

		cv::Mat_<cv::Vec3b> tmpColor;
		cv::resize(*(pm[lv - 1].getColorPtr()), tmpColor, lvSize, 0.0, 0.0, cv::INTER_LINEAR);
		
		cv::Mat_<uchar> tmpMask;
		cv::resize(*(pm[lv - 1].getMaskPtr()), tmpMask, lvSize, 0.0, 0.0, cv::INTER_LINEAR);
		for (int r = 0; r < tmpMask.rows; ++r)
		{
			uchar *ptrMask = tmpMask.ptr<uchar>(r);
			for (int c = 0; c < tmpMask.cols; ++c)
			{
				ptrMask[c] = ptrMask[c] < 255 ? 0 : 255;
			}
		}

		pm[lv].init(tmpColor, tmpMask);
	}

	// for the final composite
	mColor = color.clone();
	cv::blur(mask, mAlpha, cv::Size(blurSize, blurSize));
}

void PixMix::execute(
	cv::Mat_<cv::Vec3b> &dst,
	const float alpha
)
{
	for (int lv = int(pm.size()) - 1; lv >= 0; --lv)
	{
		pm[lv].execute(alpha, 2, 1, 0.5f);
		if (lv > 0) fillInLowerLv(pm[lv], pm[lv - 1]);
	}

	blendBorder(dst);
}

int PixMix::calcPyrmLv(
	int width,
	int height
)
{
	int size = std::min(width, height);
	int pyrmLv = 1;

	while ((size /= 2) >= 5) ++pyrmLv;

	return std::min(pyrmLv, 6);
}

void PixMix::fillInLowerLv(
	OneLvPixMix &pmUpper,
	OneLvPixMix &pmLower
)
{
	cv::Mat_<cv::Vec3b> mColorUpsampled;
	cv::resize(*(pmUpper.getColorPtr()), mColorUpsampled, pmLower.getColorPtr()->size(), 0.0, 0.0, cv::INTER_LINEAR);
	cv::Mat_<cv::Vec2i> mPosMapUpsampled;
	cv::resize(*(pmUpper.getPosMapPtr()), mPosMapUpsampled, pmLower.getPosMapPtr()->size(), 0.0, 0.0, cv::INTER_NEAREST);
	for (int r = 0; r < mPosMapUpsampled.rows; ++r)
	{
		cv::Vec2i *ptr = mPosMapUpsampled.ptr<cv::Vec2i>(r);
		for (int c = 0; c < mPosMapUpsampled.cols; ++c)
		{
			ptr[c] = ptr[c] * 2 + cv::Vec2i(r % 2, c % 2);
		}
	}

	cv::Mat_<cv::Vec3b> &mColorLw = *(pmLower.getColorPtr());
	cv::Mat_<uchar> &mMaskLw = *(pmLower.getMaskPtr());
	cv::Mat_<cv::Vec2i> &mPosMapLw = *(pmLower.getPosMapPtr());

	const int wLw = pmLower.getColorPtr()->cols;
	const int hLw = pmLower.getColorPtr()->rows;
	for (int r = 0; r < hLw; ++r)
	{
		cv::Vec3b *ptrColorLw = mColorLw.ptr<cv::Vec3b>(r);
		cv::Vec3b *ptrColorUpsampled = mColorUpsampled.ptr<cv::Vec3b>(r);
		uchar *ptrMaskLw = mMaskLw.ptr<uchar>(r);
		cv::Vec2i *ptrPosMapLw = mPosMapLw.ptr<cv::Vec2i>(r);
		cv::Vec2i *ptrPosMapUpsampled = mPosMapUpsampled.ptr<cv::Vec2i>(r);
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

void PixMix::blendBorder(
	cv::Mat_<cv::Vec3b> &dst
)
{
	cv::Mat_<cv::Vec3f> mColorF, mPMColorF, mDstF(pm[0].getColorPtr()->size());
	mColor.convertTo(mColorF, CV_32FC3, 1.0 / 255.0);
	pm[0].getColorPtr()->convertTo(mPMColorF, CV_32FC3, 1.0 / 255.0);

	cv::Mat_<float> mAlphaF;
	mAlpha.convertTo(mAlphaF, CV_32F, 1.0 / 255.0);

	for (int r = 0; r < mColor.rows; ++r)
	{
		cv::Vec3f *ptrSrc = mColorF.ptr<cv::Vec3f>(r);
		cv::Vec3f *ptrPM = mPMColorF.ptr<cv::Vec3f>(r);
		cv::Vec3f *ptrDst = mDstF.ptr<cv::Vec3f>(r);
		float *ptrAlpha = mAlphaF.ptr<float>(r);
		for (int c = 0; c < mColor.cols; ++c)
		{
			ptrDst[c] = ptrAlpha[c] * ptrSrc[c] + (1.0f - ptrAlpha[c]) * ptrPM[c];
		}
	}

	mDstF.convertTo(dst, CV_8UC3, 255.0);
}