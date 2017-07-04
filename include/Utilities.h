#ifndef _UTILS_H_
#define _UTILS_H_

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace std;
using namespace cv;

extern string modelPath;
extern string framePath;
extern Ptr<Feature2D> detector;
extern Ptr<Feature2D> descriptor;
extern NormTypes xFeatureNorm;
extern float ratio1;
extern float ratio2;

bool setupInputParameters(char **argv);
string getFrameNumber();

#endif
