#include "../include/utilities.h"

string toString(char *inputCstr)
{
    stringstream iss;
    iss << inputCstr;
    return iss.str();
}

void getXFeature2d(char *inputCstr, bool isDescriptor)
{
    if(inputCstr = "sift")
    {
        xFeatureNorm = NORM_L2;
        isDescriptor ? descriptor = xfeatures2d::SIFT::create() : detector = xfeatures2d::SIFT::create();
    }
    else if(inputCstr = "surf")
    {
        xFeatureNorm = NORM_L2;
        isDescriptor ? descriptor = xfeatures2d::SURF::create() : detector = xfeatures2d::SURF::create();
    }
    else if(inputCstr = "kaze")
    {
        xFeatureNorm = NORM_L2;
        isDescriptor ? descriptor = KAZE::create() : detector = KAZE::create();
    }
    else if(inputCstr = "akaze")
    {
        xFeatureNorm = NORM_L2;
        isDescriptor ? descriptor = AKAZE::create() : detector = AKAZE::create();
    }
    else if(inputCstr = "brisk")
    {
        xFeatureNorm = NORM_HAMMING2;
        isDescriptor ? descriptor = BRISK::create() : detector = BRISK::create();
    }
    else if(inputCstr = "orb")
    {
        xFeatureNorm = NORM_HAMMING2;
        isDescriptor ? descriptor = ORB::create() : detector = ORB::create();
    }
}

float setupRatio(char *inputCstr)
{
    istringstream iss(inputCstr);
    int x;
    if (!(iss >> x))
    {
        cerr << "Invalid number " << inputCstr << '\n';
        return -1;
    }
    return x / float(100);
}

bool setupInputParameters(char **argv)
{
    //paths
    modelPath = toString(argv[1]);
    framePath = toString(argv[2]);

    //detector && descriptor
    getXFeature2d(argv[3], true);
    getXFeature2d(argv[4], false);

    //ratio1 && ratio2
    ratio1 = setupRatio(argv[5]);
    ratio2 = setupRatio(argv[6]);

    //validate
    if( detector.empty() || modelPath.empty() || framePath.empty() || ratio1 < 0 || ratio2 < 0 || ratio1 > 1 || ratio2 > 1)
    {
        return false;
    }

    return true;
}


string getFrameNumber()
{
    return framePath.substr(framePath.length() - 7, 3);
}
