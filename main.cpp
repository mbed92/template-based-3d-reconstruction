#include "include/kpMatcher.h"
#include "include/reconstructor.h"
#include "include/utilities.h"

#include <cstdlib>

using namespace std;
using namespace cv;

string modelPath;
string framePath;
Ptr<Feature2D> detector;
Ptr<Feature2D> descriptor;
NormTypes xFeatureNorm;
float ratio1;
float ratio2;

// ./affineDSC model_name.png frame_name.png detector descriptor ratio1 ratio2
int main(int argc, char** argv)
{
    if(argc < 7 || !setupInputParameters(argv))
    {
        cerr << "Usage: ./affineDSC path_to_model.png path_to_frame.png detector descriptor ratio1% ratio2% isPointCloudSaved(0 / 1)" << endl;
        return EXIT_FAILURE;
    }

    /*describe & detect model keypoints*/
    Ptr<KpMatcher> kpm = new KpMatcher(detector, descriptor);
    Mat img = kpm->readImage(modelPath, IMREAD_GRAYSCALE );
    kpm->init(img, true);

    /*describe & detect frame keypoints*/
    Mat frame = kpm->readImage(framePath, IMREAD_GRAYSCALE );
    kpm->describeAndDetectFrameKeypoints(frame);

    /*matching*/
    kpm->findCurrentMatches(xFeatureNorm, false, ratio1);
   // kpm->improveBadMatches(xFeatureNorm, false, ratio2);

    /*visualize*/
    //kpm->drawFoundMatches(img, frame, "All matches", false);
    //kpm->drawFoundMatches(img, frame, "Only improved", true);

    /*~~~~~~~~~~~~~~~~~~~~~ 3D reconstruction (EPFL) ~~~~~~~~~~~~~~~~~~~~~~~~~*/
    Ptr<Reconstructor> rec = new Reconstructor();
    rec->init(img);

    vector<DMatch> matches = kpm->getMatches();
    vector<KeyPoint> kp1 = kpm->getModelMatchedKeypoints();
    vector<KeyPoint> kp2 = kpm->getFrameMatchedKeypoints();

    rec->prepareMatches(matches, kp1, kp2);
    rec->deform();

    cout << argv[7] << endl;
    if(argv[7] == string("1"))
    {
        stringstream ss;
        ss << "PC_" << getFrameNumber() << "_" << argv[5] << "_" << argv[6] << "_" << argv[3] << "_" << argv[4];
        rec->savePointCloud(ss.str());
        //kpm->drawFoundMatches(img, frame, "Only improved", true, ss.str());
    }

//    rec->drawMesh(img, *rec->refMesh);
//    rec->drawMesh(frame, rec->resMesh);

    rec.release();
    kpm.release();

    return EXIT_SUCCESS;
}
