#include "OneLvPixMix.h"

OneLvPixMix::OneLvPixMix()
	: borderSize(2), borderSizePosMap(1), windowSize(5), toLeft(0, -1), toRight(0, 1), toUp(-1, 0), toDown(1, 0)
{
	vSptAdj = {
		cv::Vec2i(-1, -1), cv::Vec2i(-1, 0), cv::Vec2i(-1, 1),
		cv::Vec2i( 0, -1),                   cv::Vec2i( 0, 1),
		cv::Vec2i( 1, -1), cv::Vec2i( 1, 0), cv::Vec2i( 1, 1)
	};
}

OneLvPixMix::~OneLvPixMix() { }

void OneLvPixMix::init(
	const cv::Mat_<cv::Vec3b> &color,
	const cv::Mat_<uchar> &mask
)
{
	std::random_device rnd;
	mt = std::mt19937(rnd());
	cRand = std::uniform_int_distribution<int>(0, color.cols - 1);
	rRand = std::uniform_int_distribution<int>(0, color.rows - 1);

	cv::copyMakeBorder(color, mColor[W_BORDER], borderSize, borderSize, borderSize, borderSize, cv::BORDER_REFLECT);
	cv::copyMakeBorder(mask, mMask[W_BORDER], borderSize, borderSize, borderSize, borderSize, cv::BORDER_REFLECT);
	mColor[WO_BORDER] = cv::Mat(mColor[W_BORDER], cv::Rect(borderSize, borderSize, color.cols, color.rows));
	mMask[WO_BORDER] = cv::Mat(mMask[W_BORDER], cv::Rect(borderSize, borderSize, mask.cols, mask.rows));

	mPosMap[WO_BORDER] = cv::Mat_<cv::Vec2i>(mColor[WO_BORDER].size());
	for (int r = 0; r < mPosMap[WO_BORDER].rows; ++r)
	{
		for (int c = 0; c < mPosMap[WO_BORDER].cols; ++c)
		{
			if (mMask[WO_BORDER](r, c) == 0) mPosMap[WO_BORDER](r, c) = getValidRandPos();
			else mPosMap[WO_BORDER](r, c) = cv::Vec2i(r, c);
		}
	}
	cv::copyMakeBorder(mPosMap[WO_BORDER], mPosMap[W_BORDER], borderSizePosMap, borderSizePosMap, borderSizePosMap, borderSizePosMap, cv::BORDER_REFLECT);
	mPosMap[WO_BORDER] = cv::Mat(mPosMap[W_BORDER], cv::Rect(1, 1, color.cols, color.rows));
}

void OneLvPixMix::execute(
	const float scAlpha,
	const int maxItr,
	const int maxRandSearchItr,
	const float threshDist
)
{
	const float acAlpha = 1.0f - scAlpha;
	const float thDist = std::pow(std::max(mColor[WO_BORDER].cols, mColor[WO_BORDER].rows) * threshDist, 2.0f);

	for (int itr = 0; itr < maxItr; ++itr)
	{
		cv::Mat_<cv::Vec3b> vizPosMap, vizColor;
		Util::createVizPosMap(mPosMap[WO_BORDER], vizPosMap);
		cv::resize(vizPosMap, vizPosMap, cv::Size(640, 480), 0.0, 0.0, cv::INTER_NEAREST);
		cv::imshow("prev posMap", vizPosMap);
		cv::resize(mColor[WO_BORDER], vizColor, cv::Size(640, 480), 0.0, 0.0, cv::INTER_NEAREST);
		cv::imshow("prev color", vizColor);
		cv::waitKey(1);
		
		if (itr % 2 == 0) fwdUpdate(scAlpha, acAlpha, thDist, maxRandSearchItr);
		else bwdUpdate(scAlpha, acAlpha, thDist, maxRandSearchItr);

		Util::createVizPosMap(mPosMap[WO_BORDER], vizPosMap);
		cv::resize(vizPosMap, vizPosMap, cv::Size(640, 480), 0.0, 0.0, cv::INTER_NEAREST);
		cv::imshow("curr posMap", vizPosMap);
		cv::resize(mColor[WO_BORDER], vizColor, cv::Size(640, 480), 0.0, 0.0, cv::INTER_NEAREST);
		cv::imshow("curr color", vizColor);
		cv::waitKey(1);

		inpaint();
	}
}

void OneLvPixMix::inpaint()
{
	for (int r = 0; r < mColor[WO_BORDER].rows; ++r)
	{
		cv::Vec3b *ptrColor = mColor[WO_BORDER].ptr<cv::Vec3b>(r);
		cv::Vec2i *ptrPosMap = mPosMap[WO_BORDER].ptr<cv::Vec2i>(r);
		for (int c = 0; c < mColor[WO_BORDER].cols; ++c)
		{
			ptrColor[c] = mColor[WO_BORDER](ptrPosMap[c]);
		}
	}
}

float OneLvPixMix::calcSptCost(
	const cv::Vec2i &target,
	const cv::Vec2i &ref,
	float maxDist,
	float w
)
{
	const float normFactor = maxDist * 2.0f;

	float sc = 0.0f;
	for (const auto &v : vSptAdj)
	{
		cv::Vec2f diff((ref + v) - mPosMap[W_BORDER](target + cv::Vec2i(borderSizePosMap, borderSizePosMap) + v));
		sc += std::min(diff.dot(diff), maxDist);
	}

	return sc * w / normFactor;
}

float OneLvPixMix::calcAppCost(
	const cv::Vec2i &target,
	const cv::Vec2i &ref,
	float w
)
{
	const float normFctor = 255.0f * 255.0f * 3.0f;

	float ac = 0.0f;
	for (int r = 0; r < windowSize; ++r)
	{
		uchar *ptrMask = mMask[W_BORDER].ptr<uchar>(r + ref[0]);
		cv::Vec3b *ptrTargetColor = mColor[W_BORDER].ptr<cv::Vec3b>(r + target[0]);
		cv::Vec3b *ptrRefColor = mColor[W_BORDER].ptr<cv::Vec3b>(r + ref[0]);
		for (int c = 0; c < windowSize; ++c)
		{
			if (ptrMask[c + ref[1]] == 0)
			{
				ac += FLT_MAX / 25.0f;
			}
			else
			{
				cv::Vec3f diff(cv::Vec3f(ptrTargetColor[c + target[1]]) - cv::Vec3f(ptrRefColor[c + ref[1]]));
				ac += diff.dot(diff);
			}
		}
	}

	return ac * w / normFctor;
}


void OneLvPixMix::fwdUpdate(
	const float scAlpha,
	const float acAlpha,
	const float thDist,
	const int maxRandSearchItr
)
{
#pragma omp parallel for
	for (int r = 0; r < mColor[WO_BORDER].rows; ++r)
	{
		uchar *ptrMask = mMask[WO_BORDER].ptr<uchar>(r);
		cv::Vec2i *ptrPosMap = mPosMap[WO_BORDER].ptr<cv::Vec2i>(r);
		for (int c = 0; c < mColor[WO_BORDER].cols; ++c)
		{
			if (ptrMask[c] == 0)
			{
				cv::Vec2i target(r, c);
				cv::Vec2i ref = ptrPosMap[target[1]];
				cv::Vec2i top = target + toUp;
				cv::Vec2i left = target + toLeft;
				if (top[0] < 0) top[0] = 0;
				if (left[1] < 0) left[1] = 0;
				cv::Vec2i topRef = mPosMap[WO_BORDER](top) + toDown;
				cv::Vec2i leftRef = mPosMap[WO_BORDER](left) + toRight;
				if (topRef[0] >= mColor[WO_BORDER].rows) topRef[0] = mPosMap[WO_BORDER](top)[0];
				if (leftRef[1] >= mColor[WO_BORDER].cols) leftRef[1] = mPosMap[WO_BORDER](left)[1];

				// propagate
				float cost = scAlpha * calcSptCost(target, ref, thDist) + acAlpha * calcAppCost(target, ref);
				float costTop = FLT_MAX, costLeft = FLT_MAX;

				if (mMask[WO_BORDER](top) == 0 && mMask[WO_BORDER](topRef) != 0)
				{
					costTop = scAlpha * calcSptCost(target, topRef, thDist) + acAlpha * calcAppCost(target, topRef);
				}
				if (mMask[WO_BORDER](left) == 0 && mMask[WO_BORDER](leftRef) != 0)
				{
					costLeft = scAlpha * calcSptCost(target, leftRef, thDist) + acAlpha * calcAppCost(target, leftRef);
				}

				if (costTop < cost && costTop < costLeft)
				{
					cost = costTop;
					ptrPosMap[target[1]] = topRef;
				}
				else if (costLeft < cost)
				{
					cost = costLeft;
					ptrPosMap[target[1]] = leftRef;
				}

				// random search
				int itrNum = 0;
				cv::Vec2i refRand;
				float costRand = FLT_MAX;
				do {
					refRand = getValidRandPos();
					costRand = scAlpha * calcSptCost(target, refRand, thDist) + acAlpha * calcAppCost(target, refRand);
				} while (costRand >= cost && ++itrNum < maxRandSearchItr);

				if (costRand < cost) ptrPosMap[target[1]] = refRand;
			}
		}
	}
}

void OneLvPixMix::bwdUpdate(
	const float scAlpha,
	const float acAlpha,
	const float thDist,
	const int maxRandSearchItr
)
{
#pragma omp parallel for
	for (int r = mColor[WO_BORDER].rows - 1; r >= 0; --r)
	{
		uchar *ptrMask = mMask[WO_BORDER].ptr<uchar>(r);
		cv::Vec2i *ptrPosMap = mPosMap[WO_BORDER].ptr<cv::Vec2i>(r);
		for (int c = mColor[WO_BORDER].cols - 1; c >= 0; --c)
		{
			if (ptrMask[c] == 0)
			{
				cv::Vec2i target(r, c);
				cv::Vec2i ref = ptrPosMap[target[1]];
				cv::Vec2i bottom = target + toDown;
				cv::Vec2i right = target + toRight;
				if (bottom[0] >= mColor[WO_BORDER].rows) bottom[0] = target[0];
				if (right[1] >= mColor[WO_BORDER].cols) right[1] = target[1];
				cv::Vec2i bottomRef = mPosMap[WO_BORDER](bottom) + toUp;
				cv::Vec2i rightRef = mPosMap[WO_BORDER](right) + toLeft;
				if (bottomRef[0] < 0) bottomRef[0] = 0;
				if (rightRef[1] < 0) rightRef[1] = 0;

				// propagate
				float cost = scAlpha * calcSptCost(target, ref, thDist) + acAlpha * calcAppCost(target, ref);
				float costTop = FLT_MAX, costLeft = FLT_MAX;

				if (mMask[WO_BORDER](bottom) == 0 && mMask[WO_BORDER](bottomRef) != 0)
				{
					costTop = scAlpha * calcSptCost(target, bottomRef, thDist) + acAlpha * calcAppCost(target, bottomRef);
				}
				if (mMask[WO_BORDER](right) == 0 && mMask[WO_BORDER](rightRef) != 0)
				{
					costLeft = scAlpha * calcSptCost(target, rightRef, thDist) + acAlpha * calcAppCost(target, rightRef);
				}

				if (costTop < cost && costTop < costLeft)
				{
					cost = costTop;
					ptrPosMap[target[1]] = bottomRef;
				}
				else if (costLeft < cost)
				{
					cost = costLeft;
					ptrPosMap[target[1]] = rightRef;
				}

				// random search
				int itrNum = 0;
				cv::Vec2i refRand;
				float costRand = FLT_MAX;
				do {
					refRand = getValidRandPos();
					costRand = scAlpha * calcSptCost(target, refRand, thDist) + acAlpha * calcAppCost(target, refRand);
				} while (costRand >= cost && ++itrNum < maxRandSearchItr);

				if (costRand < cost) ptrPosMap[target[1]] = refRand;
			}
		}
	}
}