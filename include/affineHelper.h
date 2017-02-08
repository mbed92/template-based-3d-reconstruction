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
    AffineHelper(Ptr<Feature2D> detect, Ptr<Feature2D> desc)
    {
        detector = detect;
        descriptor = desc;
    }

    ~AffineHelper()
    {
        free(descriptor);
        free(detector);
    }

    Mat readImage(const string &name, ImreadModes mode);
    void precomputation(Mat &img, vector<Mat> &allDescriptors, vector<vector<KeyPoint> > &allKeypoints, bool isRoi, bool isChessboardPresent);
    void findCurrentMatches(Mat &descriptors1, Mat &descriptors2, vector<DMatch> &matches, vector<DMatch> &badMatches, NormTypes norm, const float ratio);
    void drawFoundMatches(Mat &img1, vector<KeyPoint> &keypoints1, Mat &img2, vector<KeyPoint> &keypoints2, vector<DMatch> &matches, Mat &imgMatches, const string windowName);
    void improveBadMatches(vector<Mat> &descriptors1, Mat &descriptors2, vector<DMatch> &goodMatches, vector<DMatch> &badMatches, NormTypes norm, bool isFlann);

private:
    Ptr<Feature2D> detector;
    Ptr<Feature2D> descriptor;

    Mat getOnePointDescriptors(int index, vector<Mat> &descriptors);

    Point2f* setupSourcePoints(Mat &img);
    Point2f* setupDestinationPoints(Point2f *pointsSrc, TransformPoints points, int factor);
    Point2f* cpyArray(Point2f *in, int size);
    void describeAndDetectInitKeypoints(Mat &img, vector<vector<KeyPoint> > &keypoints, vector<Mat> &descriptors);
    void describeAndDetectTransformedKeypoints(Mat &transformedImg, Mat &transformMatrix, vector<KeyPoint> &initialVector, vector<vector<KeyPoint> > &keypoints, vector<Mat> &descriptors);
    void setUpRotationMatrix(Mat in, Mat &rotationMatrix, float angle, float u, float v, float w);
    void computeDescriptorsInProperOrder(vector<KeyPoint> &kp, Mat &d, Mat &img);
    void getTransformationMatrixes(const string &path, vector<Mat> &matrixes);
    void findCorners(Mat &image, vector<Point2f> &corners );
};
