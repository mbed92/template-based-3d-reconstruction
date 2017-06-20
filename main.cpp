#include "include/kpMatcher.h"
#include "include/reconstructor.h"
#include "include/utilities.h"

#include <cstdlib>

using namespace std;
using namespace cv;

/*shared variables*/
string modelPath;
string framePath;
Ptr<Feature2D> detector;
Ptr<Feature2D> descriptor;
NormTypes xFeatureNorm;
float ratio1;
float ratio2;

int main(int argc, char** argv)
{
    if(argc < 7 || !setupInputParameters(argv))
    {
        cerr << "Usage: ./affineDSC path_to_model.png path_to_frame.png detector descriptor ratio1% ratio2% isPointCloudSaved(0 / 1)" << endl;
        return EXIT_FAILURE;
    }

    /*describe & detect model keypoints*/
    Ptr<KpMatcher> kpm = new KpMatcher(detector, descriptor);
    Mat img = kpm->readImage(modelPath, IMREAD_ANYCOLOR );
    kpm->init(img);

    /*describe & detect frame keypoints*/
    Mat frame = kpm->readImage(framePath, IMREAD_ANYCOLOR );
    kpm->describeAndDetectFrameKeypoints(frame);

    /*matching*/
    kpm->findCurrentMatches(xFeatureNorm, ratio1);
    float seconds;
    clock_t t;
    t = clock();
    kpm->improveBadMatches(xFeatureNorm, ratio2);
    t = clock() - t;
    seconds = (float)t / CLOCKS_PER_SEC;

    /*~~~~~~~~~~~~~~~~~~~~~ 3D reconstruction (EPFL) ~~~~~~~~~~~~~~~~~~~~~~~~~*/
    Ptr<Reconstructor> rec = new Reconstructor();
    rec->init(img);
    vector<DMatch> matches = kpm->getMatches();
    vector<KeyPoint> kp1 = kpm->getModelMatchedKeypoints();
    vector<KeyPoint> kp2 = kpm->getFrameMatchedKeypoints();
    rec->prepareMatches(matches, kp1, kp2);
    rec->deform();
    rec->SetUseTemporal(true);
    rec->SetUsePrevFrameToInit(true);

    if(argv[7] == string("1"))
    {
        /*prepare name*/
        stringstream ss;
        ss << "PC_" << getFrameNumber() << "_" << argv[5] << "_" << argv[6] << "_" << argv[3] << "_" << argv[4];

        /*save images*/
        kpm->drawFoundMatches(img, frame, false, ss.str() + "_all");
        kpm->drawFoundMatches(img, frame, true, ss.str() + "_imp");
        kpm->drawKeypointsDisplacement(img, frame, ss.str() + "_disp");

        /*save 3d info*/
        rec->savePointCloud(ss.str());  // save point cloud
        rec->drawMesh(frame, rec->resMesh, ss.str() );  //save mesh

        /*save matches*/
        ofstream outfile;
        outfile.open("matches.txt", std::ios_base::app);
        outfile << ss.str() << " " << kpm->getMatches().size() << " " << seconds << endl;
        outfile.close();
    }

//    rec->drawMesh(img, *rec->refMesh);
//    rec->drawMesh(frame, rec->resMesh);

    rec.release();
    kpm.release();

    return EXIT_SUCCESS;
}
