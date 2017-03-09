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
    KpMatcher* kpm = new KpMatcher(detector, descriptor);
    Mat img = kpm->readImage("images/model.png", IMREAD_GRAYSCALE );
    kpm->init(img, true);

    /*describe & detect frame keypoints*/
    Mat frame = kpm->readImage("images/frame_128.png", IMREAD_GRAYSCALE );
    kpm->describeAndDetectFrameKeypoints(frame);

    /*matching*/
    kpm->findCurrentMatches(NORM_L2, true, 0.8);
    kpm->improveBadMatches(NORM_L2, true, 0.7);

    /*visualize*/
    kpm->drawFoundMatches(img, frame, "All matches", false);
    kpm->drawFoundMatches(img, frame, "Only improved", true);

    //    /*~~~~~~~~~~~~~~~~~~~~~ 3D reconstruction (EPFL) ~~~~~~~~~~~~~~~~~~~~~~~~~*/
    //    Reconstructor *rec = new Reconstructor();
    //    rec->init();
    //    rec->drawRefMesh(img);

    delete kpm;
    return EXIT_SUCCESS;
}
