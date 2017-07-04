#include "../include/KpMatcher.h"
#include <opencv2/opencv.hpp>

//
// public section
//
KpMatcher::KpMatcher(cv::Ptr<cv::Feature2D> detect, cv::Ptr<cv::Feature2D> desc)
{
    this->isSimulatedLocalDeform = true;

    detector = detect;
    descriptor = desc;

    //TODO read ROI from file
    cv::Point2f topLeft = cv::Point2f(227, 100);
    cv::Point2f topRight = cv::Point2f(515, 122);
    cv::Point2f bottomRight = cv::Point2f(506, 367);
    cv::Point2f bottomLeft = cv::Point2f(208, 343);
    parallellogram = {bottomLeft, topLeft, topRight, bottomRight, bottomLeft};
}

KpMatcher::~KpMatcher()
{
    detector.release();
    descriptor.release();
}

cv::Mat KpMatcher::ReadImage(const std::string &name, cv::ImreadModes mode)
{
    cv::Mat img = imread(name, mode);
    if ( img.empty() )
    {
        std::cerr << "Error while loading an image " << name << std::endl;
        abort();
    }
    resize(img, img, cv::Size(640, 480));
    std::cout << "Image " << name << " properly loaded" << std::endl;
    return img;
}


void KpMatcher::Init(const cv::Mat &img)
{
    std::cout << "Compute descriptors for input image." << std::endl;
    this->DescribeAndDetectInitKeypoints(img);

    if(this->isSimulatedLocalDeform)
    {
        std::cout << "Initialization started..." << std::endl;
        this->GetTransformationMatrixes("./chessboard.txt");

        for(size_t i = 0; i < this->transformationMatrixes.size(); i++)
        {
            cv::Mat dst = cv::Mat::zeros( img.rows, img.cols, img.type() );
            cv::warpPerspective( img, dst, this->transformationMatrixes[i], cv::Size(img.cols, img.rows));

            std::cout << "Compute descriptors for affine transformation: " << i << std::endl;
            this->DescribeAndDetectTransformedKeypoints(dst, this->transformationMatrixes[i]);
        }
    }
    std::cout << "Initialization finished." << std::endl;
}

void KpMatcher::DescribeAndDetectFrameKeypoints(const cv::Mat& frame)
{
    this->detector->detect(frame, keypointsFrame);
    this->descriptor->compute( frame, keypointsFrame, descriptorsFrame );
}

void KpMatcher::FindCurrentMatches(cv::NormTypes norm, const float &ratio)
{
    cv::DescriptorMatcher* matcher = new cv::BFMatcher(norm);

    std::vector<std::vector<cv::DMatch> > potentialMatches;
    matcher->knnMatch(this->descriptors[0], this->descriptorsFrame, potentialMatches, 2);
    std::cout << potentialMatches.size() << std::endl;

    for (size_t i = 0; i < potentialMatches.size(); ++i)
    {
        if (potentialMatches[i][0].distance < ratio * potentialMatches[i][1].distance)
        {
            this->goodMatches.push_back(potentialMatches[i][0]);
        }
        else
        {
            this->badMatches.push_back(potentialMatches[i][0]);
        }
    }
}

cv::Mat KpMatcher::GetOnePointDescriptors(int index)
{
    cv::Mat temp;

    for(size_t i=1;i<this->descriptors.size();i++)
    {
        if(index < this->descriptors[i].rows)
        {
            temp.push_back(this->descriptors[i].row(index));
        }
    }
    return temp;
}

void KpMatcher::ImproveBadMatches(cv::NormTypes norm, const float &ratio)
{
    if(this->isSimulatedLocalDeform)
    {
        std::cout << "RATIO: " << ratio << std::endl;
        cv::DescriptorMatcher* matcher = new cv::BFMatcher(norm);

        std::vector<std::vector<cv::DMatch> > potentialMatches;

        std::cout << "Improve the quality of matching using affine transformations..." << std::endl;
        for(size_t i = 0; i < this->badMatches.size(); i++)
        {
            int originalFotoIdx = this->badMatches[i].queryIdx;
            int comparedFotoIdx = this->badMatches[i].trainIdx;
            float originalDistance = this->badMatches[i].distance;
            cv::Mat onePointDesc = this->GetOnePointDescriptors(originalFotoIdx);

            if(!onePointDesc.empty())
            {
                cv::Mat desc2;
                cv::KeyPoint to = this->keypointsFrame[comparedFotoIdx];
                cv::KeyPoint from = this->keypoints[0][originalFotoIdx];
                double dist = sqrt((from.pt.x - to.pt.x) * (from.pt.x - to.pt.x) + (from.pt.y - to.pt.y) * (from.pt.y - to.pt.y));
                desc2.push_back(this->descriptorsFrame.row(comparedFotoIdx));
                matcher->knnMatch( desc2, onePointDesc, potentialMatches, 2 );
                if(dist < 30 && !potentialMatches.empty() && potentialMatches[0][0].distance < ratio * potentialMatches[0][1].distance && potentialMatches[0][0].distance < originalDistance )
                {
                    badMatches[i].distance = potentialMatches[0][0].distance;
                    this->goodMatches.push_back(this->badMatches[i]);
                    this->improvementMatches.push_back(this->badMatches[i]);
                }
                potentialMatches.clear();
                onePointDesc.release();
                desc2.release();
            }
        }
    }
}

void KpMatcher::DrawKeypointsDisplacement(const cv::Mat& img1, const cv::Mat& img2, std::string fileName)
{
    fileName = fileName + ".png";
    cv::Mat temp1, temp2;
    img1.copyTo(temp1);
    img2.copyTo(temp2);
    if(!this->improvementMatches.empty() )
    {
        for(size_t i = 0; i < this->improvementMatches.size(); ++i)
        {
            cv::KeyPoint to = this->keypointsFrame[this->improvementMatches[i].trainIdx];
            cv::KeyPoint from = this->keypoints[0][this->improvementMatches[i].queryIdx];
            line(temp2, from.pt, to.pt, cv::Scalar(0, 0, 255), 2, 8, 0);
        }
        std::cout << "Size of improved matches: " <<  this->improvementMatches.size() << std::endl;
    }
    else
    {
        std::cerr << "Cannot show matches. Check if you performed improvement matches process." << std::endl;
        return;
    }
    imwrite( fileName, temp2 );
}

void KpMatcher::DrawFoundMatches(const cv::Mat& img1, const cv::Mat& img2, bool drawOnlyImproved, std::string fileName)
{
    fileName = fileName + ".png";
    cv::Mat temp1, temp2, imgMatches;
    img1.copyTo(temp1);
    img2.copyTo(temp2);
    if(drawOnlyImproved && !this->improvementMatches.empty() )
    {
        drawMatches(temp1, this->keypoints[0], temp2, this->keypointsFrame, this->improvementMatches, imgMatches);
        std::cout << "Size of improved matches: " <<  this->improvementMatches.size() << std::endl;
    }
    else if(!drawOnlyImproved)
    {
        drawMatches(temp1, this->keypoints[0], temp2, this->keypointsFrame, this->goodMatches, imgMatches);
        std::cout << "Size of all keypoints: " <<  this->keypoints[0].size() << std::endl;
        std::cout << "Size of all matches: " <<  this->goodMatches.size() << std::endl;
    }
    else
    {
        std::cerr << "Cannot show matches. Check if you performed improvement matches process." << std::endl;
        return;
    }
    imwrite( fileName, imgMatches );
}

std::vector<cv::DMatch> KpMatcher::GetMatches()
{
    return this->goodMatches;
}

std::vector<cv::KeyPoint> KpMatcher::GetModelMatchedKeypoints()
{
    return this->keypoints[0];
}

std::vector<cv::KeyPoint> KpMatcher::GetFrameMatchedKeypoints()
{
    return this->keypointsFrame;
}



//
//private section
//

bool KpMatcher::IsInParalellgoram(cv::KeyPoint& kp)
{
    //couter-clockwise
    for(int i = this->parallellogram.size(); i > 0; i--)
    {
        float A = -(this->parallellogram[i-1].y - this->parallellogram[i].y);
        float B = this->parallellogram[i-1].x - this->parallellogram[i].x;
        float C = -(A * this->parallellogram[i].x + B * this->parallellogram[i].y);
        float D = A * kp.pt.x + B * kp.pt.y + C;
        if(D > 0)
        {
            return false;
        }
    }
    return true;
}

void KpMatcher::DescribeAndDetectInitKeypoints(const cv::Mat &img)
{
    std::vector<cv::KeyPoint> keypointsPre;
    detector->detect(img, keypointsPre);
    for(size_t i = 0; i < keypointsPre.size(); i++)
    {
        if(!this->IsInParalellgoram(keypointsPre[i]))
        {
            keypointsPre.erase(keypointsPre.begin() + i);
            i--;
        }
    }
    this->keypoints.push_back(keypointsPre);

    cv::Mat desc;
    this->descriptor->compute( img, keypointsPre, desc );
    this->descriptors.push_back(desc);

    keypointsPre.clear();
    desc.release();
}

void KpMatcher::DescribeAndDetectTransformedKeypoints(const cv::Mat& transformedImg, const cv::Mat& transformMatrix)
{
    std::vector<cv::KeyPoint> initialVector(this->keypoints[0]);
    std::vector<cv::KeyPoint> keypointsPre(initialVector);
    std::vector<cv::Point2f> kp;
    for(size_t i = 0; i < initialVector.size(); i++)
    {
        kp.push_back(initialVector[i].pt);
    }
    cv::perspectiveTransform(kp, kp, transformMatrix);
    for(size_t i = 0; i < keypointsPre.size(); i++)
    {
        keypointsPre[i].pt = kp[i];
    }
    this->keypoints.push_back(keypointsPre);

    cv::Mat desc;
    this->descriptor->compute( transformedImg, keypointsPre, desc );
    this->descriptors.push_back(desc);
}

void KpMatcher::FindCorners(const cv::Mat &image, std::vector<cv::Point2f> &corners )
{
    //TODO read chessboard size from file
    cv::Size pattern = cv::Size(9,6);
    bool isFound = findChessboardCorners(image, pattern, corners);
    if(!isFound)
    {
        std::cerr << "Error occured while procssing chessboard data." << std::endl;
        abort();
    }
}

void KpMatcher::GetTransformationMatrixes(const std::string &path)
{
    std::ifstream infile(path.c_str());
    std::vector<cv::Point2f> initialCorners;
    int i = 0;
    std::string line;

    while (getline(infile, line))
    {
        if(i > 0)
        {
            std::vector<cv::Point2f> nextCorners;
            cv::Mat nextImage = ReadImage(line, cv::IMREAD_ANYCOLOR);
            this->FindCorners(nextImage, nextCorners);
            cv::Mat homographyMatrix = findHomography(initialCorners, nextCorners, cv::RANSAC);
            this->transformationMatrixes.push_back(homographyMatrix);
        }
        else
        {
            cv::Mat initialImage = ReadImage(line, cv::IMREAD_ANYCOLOR);
            this->FindCorners(initialImage, initialCorners);
        }
        i++;
    }
    infile.close();
}
