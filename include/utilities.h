#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <fstream>
#include <sstream>
#include <string>

#define NPOINTS 4

using namespace std;
using namespace cv;

enum TransformPoints
{
    Top,
    TopRight,
    Right,
    LowRight,
    Low,
    LowLeft,
    Left,
    TopLeft,
    First = Top,
    Last = TopLeft
};

void CallBackFunc(int event, int x, int y, int flags, void* userdata);


