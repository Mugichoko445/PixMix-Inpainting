#include <iostream>
#include "PixMix/PixMix.h"

//#define MAGENTA_MASK_MODE

#ifndef MAGENTA_MASK_MODE
const std::string pathToSrcColor("../data/birds.png");
const std::string pathToDstColor("../data/birds_res.png");
const std::string pathToMask("../data/birds_mask.png");

int main()
{
	cv::Mat_<cv::Vec3b> src = cv::imread(pathToSrcColor);
	cv::Mat_<cv::Vec3b> dst(src.size());
	cv::Mat_<uchar> mask = cv::imread(pathToMask, cv::IMREAD_GRAYSCALE);
	
	cv::imshow("Input color image", src);
	cv::imshow("Input mask image", mask);
	cv::waitKey(1);

	PixMix pm;
	pm.init(src, mask);
	
	pm.execute(dst, 0.05f);

	cv::imwrite(pathToDstColor, dst);
	cv::imshow("Output color image", dst);
	cv::waitKey();

	return 0;
}

#else
const std::string pathToSrcColor("../data/birds_magenta.png");
const std::string pathToDstColor("../data/birds_magenta_res.png");

int main()
{
	cv::Mat_<cv::Vec3b> src = cv::imread(pathToSrcColor);
	cv::Mat_<cv::Vec3b> dst(src.size(), src.type());
	cv::Mat_<uchar> mask;
	
	Util::createMask(src, cv::Scalar(255, 0, 255), mask);

	cv::imshow("Input color image", src);
	cv::imshow("Input mask image", mask);
	cv::waitKey(1);

	PixMix pm;
	pm.init(src, mask);

	pm.execute(dst, 0.05f);

	cv::imshow("Output color image", dst);
	cv::waitKey();

	return 0;
}
#endif