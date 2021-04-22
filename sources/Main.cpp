#include <iostream>
#include "PixMix/PixMix.h"

const std::string pathToSrcColor("../../data/birds.png");
const std::string pathToDstColor("../../data/birds_res.png");
const std::string pathToMask("../../data/birds_mask.png");

int main()
{
	auto color = cv::imread(pathToSrcColor, cv::IMREAD_COLOR);
	auto mask = cv::imread(pathToMask, cv::IMREAD_GRAYSCALE);
	cv::Mat inpainted;

	cv::imshow("Input color image", color);
	cv::imshow("Input mask image", mask);
	cv::waitKey();
	cv::destroyAllWindows();

	auto start = std::chrono::high_resolution_clock::now();
	{
		dr::PixMix pm;
		dr::det::PixMixParams params;
		params.alpha = 0.15f;			// 0.0f means no spatial cost considered
		params.maxItr = 5;				// set to 1 to crank up the speed
		params.maxRandSearchItr = 5;	// set to 1 to crank up the speed
		bool debugViz = true;
		pm.Run(color, mask, inpainted, params, debugViz);
	}
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "Time: " << duration.count() << "ms" << std::endl;

	cv::imwrite(pathToDstColor, inpainted);
	cv::imshow("Inpainted color image", inpainted);
	cv::waitKey();

	return 0;
}
