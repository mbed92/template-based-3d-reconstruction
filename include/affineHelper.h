#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "utilities.h"

using namespace std;
using namespace cv;

class AffineHelper
{
public:
    AffineHelper(Ptr<Feature2D> f2d)
    {
        alghoritm = f2d;
    }

    ~AffineHelper()
    {
        free(alghoritm);
    }

    Mat readImage(const string& name, ImreadModes mode);
    void precomputation(Mat& img, vector<Mat> &allDescriptors, vector<vector<KeyPoint> > &allKeypoints);
    void findCurrentMatches(Mat& descriptors1, Mat& descriptors2, vector<DMatch>& matches, NormTypes norm, const float ratio);
    void drawFoundMatches(Mat& img1, vector<KeyPoint>& keypoints1, Mat& img2, vector<KeyPoint>& keypoints2, vector<DMatch>& matches, Mat& imgMatches);

private:
    Ptr<Feature2D> alghoritm;

    Point2f* setupSourcePoints(Mat &img);
    Point2f* setupDestinationPoints(Point2f *pointsSrc, TransformPoints points, int factor);
    Point2f* cpyArray(Point2f* in, int size);
    void describeAndDetect(Mat &img, vector<vector<KeyPoint> > &keypoints, vector<Mat> &descriptors);
};
