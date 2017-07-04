#include "../include/Utilities.h"

std::string ToString(char *inputCstr)
{
    std::stringstream iss;
    iss << inputCstr;
    return iss.str();
}

void GetXFeature2d(char *inputCstr, bool isDescriptor)
{
    if(inputCstr == std::string("sift"))
    {
        if(isDescriptor)
        {
            std::cout << "Descriptor SIFT choosed." << std::endl;
            xFeatureNorm = cv::NORM_L2;
            descriptor = cv::xfeatures2d::SIFT::create();
        }
        else
        {
            std::cout << "Detector SIFT choosed." << std::endl;
            detector = cv::xfeatures2d::SIFT::create();
        }
    }
    else if(inputCstr == std::string("surf"))
    {
        if(isDescriptor)
        {
            std::cout << "Descriptor SURF choosed." << std::endl;
            xFeatureNorm = cv::NORM_L2;
            descriptor = cv::xfeatures2d::SURF::create();
        }
        else
        {
            std::cout << "Detector SURF choosed." << std::endl;
            detector = cv::xfeatures2d::SURF::create();
        }
    }
    else if(inputCstr == std::string("kaze"))
    {
        if(isDescriptor)
        {
            std::cout << "Descriptor KAZE choosed." << std::endl;
            xFeatureNorm = cv::NORM_L2;
            descriptor = cv::KAZE::create();
        }
        else
        {
            std::cout << "Detector KAZE choosed." << std::endl;
            detector = cv::KAZE::create();
        }
    }
    else if(inputCstr == std::string("akaze"))
    {
        if(isDescriptor)
        {
            std::cout << "Descriptor AKAZE choosed." << std::endl;
            xFeatureNorm = cv::NORM_HAMMING2;
            descriptor = cv::AKAZE::create();
        }
        else
        {
            std::cout << "Detector AKAZE choosed." << std::endl;
            detector = cv::AKAZE::create();
        }
    }
    else if(inputCstr == std::string("brisk"))
    {
        if(isDescriptor)
        {
            std::cout << "Descriptor BRISK choosed." << std::endl;
            xFeatureNorm = cv::NORM_HAMMING2;
            descriptor = cv::BRISK::create();
        }
        else
        {
            std::cout << "Detector BRISK choosed." << std::endl;
            detector = cv::BRISK::create();
        }
    }
    else if(inputCstr == std::string("orb"))
    {
        if(isDescriptor)
        {
            std::cout << "Descriptor ORB choosed." << std::endl;
            xFeatureNorm = cv::NORM_HAMMING2;
            descriptor = cv::ORB::create();
        }
        else
        {
            std::cout << "Detector ORB choosed." << std::endl;
            detector = cv::ORB::create();
        }
    }
}

float SetupRatio(char *inputCstr)
{
    std::istringstream iss(inputCstr);
    int x;
    if (!(iss >> x))
    {
        std::cerr << "Invalid number " << inputCstr << '\n';
        return -1;
    }
    return x / float(100);
}

bool SetupInputParameters(char **argv)
{
    //paths
    modelPath = ToString(argv[1]);
    framePath = ToString(argv[2]);

    //detector && descriptor
    GetXFeature2d(argv[3], false);
    GetXFeature2d(argv[4], true);

    //ratio1 && ratio2
    ratio1 = SetupRatio(argv[5]);
    ratio2 = SetupRatio(argv[6]);

    //validate
    if( detector.empty() || descriptor.empty() || modelPath.empty() || ratio1 < 0 || ratio2 < 0 || ratio1 > 1 || ratio2 > 1)
    {
        return false;
    }

    return true;
}


std::string GetFrameNumber()
{
    return framePath.substr(framePath.length() - 7, 3);
}
