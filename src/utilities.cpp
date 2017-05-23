#include "../include/utilities.h"

string toString(char *inputCstr)
{
    stringstream iss;
    iss << inputCstr;
    return iss.str();
}

void getXFeature2d(char *inputCstr, bool isDescriptor)
{
    if(inputCstr == string("sift"))
    {
        if(isDescriptor)
        {
            cout << "Descriptor SIFT choosed." << endl;
            xFeatureNorm = NORM_L2;
            descriptor = xfeatures2d::SIFT::create();
        }
        else
        {
            cout << "Detector SIFT choosed." << endl;
            detector = xfeatures2d::SIFT::create();
        }
    }
    else if(inputCstr == string("surf"))
    {
        if(isDescriptor)
        {
            cout << "Descriptor SURF choosed." << endl;
            xFeatureNorm = NORM_L2;
            descriptor = xfeatures2d::SURF::create();
        }
        else
        {
            cout << "Detector SURF choosed." << endl;
            detector = xfeatures2d::SURF::create();
        }
    }
    else if(inputCstr == string("kaze"))
    {
        if(isDescriptor)
        {
            cout << "Descriptor KAZE choosed." << endl;
            xFeatureNorm = NORM_L2;
            descriptor = KAZE::create();
        }
        else
        {
            cout << "Detector KAZE choosed." << endl;
            detector = KAZE::create();
        }
    }
    else if(inputCstr == string("akaze"))
    {
        if(isDescriptor)
        {
            cout << "Descriptor AKAZE choosed." << endl;
            xFeatureNorm = NORM_HAMMING2;
            descriptor = AKAZE::create();
        }
        else
        {
            cout << "Detector AKAZE choosed." << endl;
            detector = AKAZE::create();
        }
    }
    else if(inputCstr == string("brisk"))
    {
        if(isDescriptor)
        {
            cout << "Descriptor BRISK choosed." << endl;
            xFeatureNorm = NORM_HAMMING2;
            descriptor = BRISK::create();
        }
        else
        {
            cout << "Detector BRISK choosed." << endl;
            detector = BRISK::create();
        }
    }
    else if(inputCstr == string("orb"))
    {
        if(isDescriptor)
        {
            cout << "Descriptor ORB choosed." << endl;
            xFeatureNorm = NORM_HAMMING2;
            descriptor = ORB::create();
        }
        else
        {
            cout << "Detector ORB choosed." << endl;
            detector = ORB::create();
        }
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
    getXFeature2d(argv[3], false);
    getXFeature2d(argv[4], true);

    //ratio1 && ratio2
    ratio1 = setupRatio(argv[5]);
    ratio2 = setupRatio(argv[6]);

    //validate
    if( detector.empty() || descriptor.empty() || modelPath.empty() || ratio1 < 0 || ratio2 < 0 || ratio1 > 1 || ratio2 > 1)
    {
        return false;
    }

    return true;
}


string getFrameNumber()
{
    return framePath.substr(framePath.length() - 7, 3);
}
