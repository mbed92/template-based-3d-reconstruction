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

        //TODO read ROI from file
        Point2f topLeft = Point2f(227, 100);
        Point2f topRight = Point2f(515, 122);
        Point2f bottomRight = Point2f(506, 367);
        Point2f bottomLeft = Point2f(208, 343);
        parallellogram.push_back(bottomLeft);
        parallellogram.push_back(topLeft);
        parallellogram.push_back(topRight);
        parallellogram.push_back(bottomRight);
        parallellogram.push_back(bottomLeft);
    }

    ~KpMatcher()
    {
        detector.release();
        descriptor.release();
    }

    /*ASIFT++*/
    Mat readImage(const string &name, ImreadModes mode);
    void init(const Mat& img);
    void describeAndDetectFrameKeypoints(const Mat& frame);
    void findCurrentMatches(NormTypes norm, bool isFlann, const float& ratio);
    void improveBadMatches(NormTypes norm, bool isFlann, const float& ratio);
    void drawFoundMatches(const Mat &img1, const Mat &img2, bool drawOnlyImproved, string fileName);
    void drawKeypointsDisplacement(const Mat &img1, const Mat &img2, string fileName);

    /*getters*/
    vector<DMatch> getMatches();
    vector<KeyPoint> getModelMatchedKeypoints();
    vector<KeyPoint> getFrameMatchedKeypoints();

private:
    vector<Point2f> parallellogram;

    Mat getOnePointDescriptors(int index);
    bool isInParalellgoram(KeyPoint& kp);
    void describeAndDetectInitKeypoints(const Mat& img);
    void describeAndDetectTransformedKeypoints(const Mat& transformedImg, const Mat& transformMatrix);
    void getTransformationMatrixes(const string& path);
    void findCorners(const Mat& image, vector<Point2f>& corners );
};
#endif
