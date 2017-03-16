#include "include/kpMatcher.h"
#include "include/reconstructor.h"

using namespace std;
using namespace cv;

int main()
{
    /*init data containers*/
    Ptr<Feature2D> detector = xfeatures2d::SIFT::create();
    Ptr<Feature2D> descriptor = xfeatures2d::SIFT::create();

    /*describe & detect model keypoints*/
    Ptr<KpMatcher> kpm = new KpMatcher(detector, descriptor);
    Mat img = kpm->readImage("images/model.png", IMREAD_GRAYSCALE );
    kpm->init(img, true);

    /*describe & detect frame keypoints*/
    Mat frame = kpm->readImage("images/frame_128.png", IMREAD_GRAYSCALE );
    kpm->describeAndDetectFrameKeypoints(frame);

    /*matching*/
    kpm->findCurrentMatches(NORM_L2, true, 0.8);
    //kpm->improveBadMatches(NORM_L2, true, 0.7);

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
    //rec->openGLproj();
    rec->drawMesh(frame);





    kpm.release();
    rec.release();
    return EXIT_SUCCESS;
}
