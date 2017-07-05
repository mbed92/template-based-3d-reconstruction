//////////////////////////////////////////////////////////////////////////
// Author		:	Michał Bednarek
// Email		:	michal.gr.bednarek@doctorate.put.poznan.pl
// Organization	:	Poznan University of Technology
// Date			:	2017
//////////////////////////////////////////////////////////////////////////

#ifndef _KPMATCHER_
#define _KPMATCHER_

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>

#include <time.h>
#include <ctime>

#define NPOINTS 4

class KpMatcher
{
public:
    KpMatcher(cv::Ptr<cv::Feature2D> detect, cv::Ptr<cv::Feature2D> desc);
    ~KpMatcher();

    cv::Mat ReadImage(const std::string &name, cv::ImreadModes mode);
    void Init(const cv::Mat& img);
    void DescribeAndDetectFrameKeypoints(const cv::Mat& frame);
    void FindCurrentMatches(cv::NormTypes norm, const float& ratio);
    void ImproveBadMatches(cv::NormTypes norm, const float& ratio);
    void DrawFoundMatches(const cv::Mat &img1, const cv::Mat &img2, bool drawOnlyImproved, std::string fileName);
    void DrawKeypointsDisplacement(const cv::Mat &img1, const cv::Mat &img2, std::string fileName);
    std::vector<cv::DMatch> GetMatches();
    std::vector<cv::KeyPoint> GetModelMatchedKeypoints();
    std::vector<cv::KeyPoint> GetFrameMatchedKeypoints();

private:
    bool isSimulatedLocalDeform;
    std::vector<cv::Mat> descriptors;
    std::vector<std::vector<cv::KeyPoint> > keypoints;
    cv::Mat descriptorsFrame;
    std::vector<cv::KeyPoint> keypointsFrame;
    std::vector<cv::DMatch> goodMatches, badMatches, improvementMatches;
    std::vector<cv::Mat> transformationMatrixes;
    cv::Ptr<cv::Feature2D> detector;
    cv::Ptr<cv::Feature2D> descriptor;
    std::vector<cv::Point2f> parallellogram;

private:
    cv::Mat GetOnePointDescriptors(int index);
    bool IsInParalellgoram(cv::KeyPoint& kp);
    void DescribeAndDetectInitKeypoints(const cv::Mat& img);
    void DescribeAndDetectTransformedKeypoints(const cv::Mat& transformedImg, const cv::Mat& transformMatrix);
    void GetTransformationMatrixes(const std::string& path);
    void FindCorners(const cv::Mat& image, std::vector<cv::Point2f>& corners );
};
#endif
