#pragma once

#include <random>
#include <opencv2/opencv.hpp>

#include "Utilities.h"

class OneLvPixMix
{
public:
	OneLvPixMix();
	~OneLvPixMix();

	void init(const cv::Mat_<cv::Vec3b> &color, const cv::Mat_<uchar> &mask);
	void execute(const float scAlpha, const int maxItr, const int maxRandSearchItr, const float threshDist);

	cv::Mat_<cv::Vec3b> *getColorPtr();
	cv::Mat_<uchar> *getMaskPtr();
	cv::Mat_<cv::Vec2i> *getPosMapPtr();

private:
	const int borderSize;
	const int borderSizePosMap;
	const int windowSize;

	enum { WO_BORDER = 0, W_BORDER = 1 };
	cv::Mat_<cv::Vec3b> mColor[2];
	cv::Mat_<uchar> mMask[2];
	cv::Mat_<cv::Vec2i> mPosMap[2];	// current position map: f

	const cv::Vec2i toLeft;
	const cv::Vec2i toRight;
	const cv::Vec2i toUp;
	const cv::Vec2i toDown;
	std::vector<cv::Vec2i> vSptAdj;

	std::mt19937 mt;
	std::uniform_int_distribution<int> cRand;
	std::uniform_int_distribution<int> rRand;

	cv::Vec2i getValidRandPos();

	void inpaint();

	float calcSptCost(
		const cv::Vec2i &target,
		const cv::Vec2i &ref,
		float maxDist,		// tau_s
		float w = 0.125f	// 1.0f / 8.0f
	);
	float calcAppCost(
		const cv::Vec2i &target,
		const cv::Vec2i &ref,
		float w = 0.04f		// 1.0f / 25.0f
	);

	void fwdUpdate(
		const float scAlpha,
		const float acAlpha,
		const float thDist,
		const int maxRandSearchItr
	);
	void bwdUpdate(
		const float scAlpha,
		const float acAlpha,
		const float thDist,
		const int maxRandSearchItr
	);
};

inline cv::Mat_<cv::Vec3b> *OneLvPixMix::getColorPtr()
{
	return &(mColor[WO_BORDER]);
}
inline cv::Mat_<uchar> *OneLvPixMix::getMaskPtr()
{
	return &(mMask[WO_BORDER]);
}
inline cv::Mat_<cv::Vec2i> *OneLvPixMix::getPosMapPtr()
{
	return &mPosMap[WO_BORDER];
}

inline cv::Vec2i OneLvPixMix::getValidRandPos()
{
	cv::Vec2i p;
	do {
		p = cv::Vec2i(rRand(mt), cRand(mt));
	} while (mMask[WO_BORDER](p) != 255);

	return p;
}