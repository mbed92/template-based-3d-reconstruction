//////////////////////////////////////////////////////////////////////////
// Author		:	Michał Bednarek
// Email		:	michal.gr.bednarek@doctorate.put.poznan.pl
// Organization	:	Poznan University of Technology
// Date			:	2017
//////////////////////////////////////////////////////////////////////////

#ifndef _UTILS_H_
#define _UTILS_H_

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

extern std::string modelPath;
extern std::string framePath;
extern cv::Ptr<cv::Feature2D> detector;
extern cv::Ptr<cv::Feature2D> descriptor;
extern cv::NormTypes xFeatureNorm;
extern float ratio1;
extern float ratio2;

bool SetupInputParameters(char **argv);
std::string GetFrameNumber();

#endif
