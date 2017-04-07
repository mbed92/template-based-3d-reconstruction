//////////////////////////////////////////////////////////////////////////
// Author		:	Michał Bednarek
// Email		:	michal.gr.bednarek@doctorate.put.poznan.pl
// Organization	:	Poznan University of Technology
// Purpose		:	Improvement in ASIFT alghoritm
// Date			:	in progress since Jan 2017
//////////////////////////////////////////////////////////////////////////

#ifndef _KPMATCHER_
#define _KPMATCHER_


#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d.hpp"

#include <GLFW/glfw3.h>
#include <GL/gl.h>
#include <GL/glu.h>

#include <time.h>
#include <ctime>

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

class KpMatcher
{
private:
    vector<Mat> descriptors;
    vector<vector<KeyPoint> > keypoints;
    Mat descriptorsFrame;
    vector<KeyPoint> keypointsFrame;
    vector<DMatch> goodMatches, badMatches, improvementMatches;
    vector<Mat> transformationMatrixes;
    Ptr<Feature2D> detector;
    Ptr<Feature2D> descriptor;

public:
    KpMatcher(Ptr<Feature2D> detect, Ptr<Feature2D> desc)
    {
        detector = detect;
        descriptor = desc;
    }

    ~KpMatcher()
    {
        detector.release();
        descriptor.release();
    }

    /*ASIFT++*/
    Mat readImage(const string &name, ImreadModes mode);
    void init(Mat &img, bool isChessboardPresent);
    void describeAndDetectFrameKeypoints(Mat &frame, float &seconds);
    void findCurrentMatches(NormTypes norm, bool isFlann, const float &ratio);
    void improveBadMatches(NormTypes norm, bool isFlann, const float &ratio);
    void drawFoundMatches(Mat &img1, Mat &img2, const string &windowName, bool drawOnlyImproved);

    /*getters*/
    vector<DMatch> getMatches();
    vector<KeyPoint> getModelMatchedKeypoints();
    vector<KeyPoint> getFrameMatchedKeypoints();


    /*To 3d visualize - under development*/
    void getReprojectionMatrixes(const string &intr, const string &extr, const string &k, Mat &intrinsticMatrix, Mat &extrinsicMatrix, Mat &kMatrix);
    void get2dPointsFromKeypoints(vector<KeyPoint> &keypointsFrame, vector<Vec3f> &points2d);
    void getSpatialCoordinates(Mat &intr, Mat &extr, vector<Vec3f> &points2d, vector<Vec3f> &points3d);
    void pcshow(vector<Vec3f> &points3d);

private:
    void readMatrixFromFile(const string& path, const int cols, Mat &out);
    Mat getOnePointDescriptors(int index);
    bool isInParalellgoram(KeyPoint kp, vector<Point2f> &figure);
    void describeAndDetectInitKeypoints(Mat &img);
    void describeAndDetectTransformedKeypoints(Mat &transformedImg, Mat &transformMatrix);
    void setUpRotationMatrix(Mat in, Mat &rotationMatrix, float angle, float u, float v, float w);
    void computeDescriptorsInProperOrder(vector<KeyPoint> &kp, Mat &d, Mat &img);
    void getTransformationMatrixes(const string &path);
    void findCorners(Mat &image, vector<Point2f> &corners );

    Point2f* setupSourcePoints(Mat &img);
    Point2f* setupDestinationPoints(Point2f *pointsSrc, TransformPoints points, int factor);
    Point2f* cpyArray(Point2f *in, int size);
};
#endif
