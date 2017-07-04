//////////////////////////////////////////////////////////////////////////
// Author		:	Michał Bednarek
// Email		:	michal.gr.bednarek@doctorate.put.poznan.pl
// Organization	:	Poznan University of Technology
// Date			:	2017
//////////////////////////////////////////////////////////////////////////

#include "include/KpMatcher.h"
#include "include/Reconstructor.h"
#include "include/Utilities.h"
#include "include/SimulatedAnnealing.h"

#include <memory>
#include <cstdlib>

/*shared variables  */
string modelPath;
string framePath;
Ptr<Feature2D> detector;
Ptr<Feature2D> descriptor;
NormTypes xFeatureNorm;
float ratio1;
float ratio2;

int main(int argc, char** argv)
{
    if(argc < 7 || !SetupInputParameters(argv))
    {
        cerr << "Usage: ./affineDSC path_to_model.png path_to_frame.png detector descriptor ratio1% ratio2% isPointCloudSaved(0 / 1)" << endl;
        return EXIT_FAILURE;
    }

    /*describe & detect model keypoints*/
    unique_ptr<KpMatcher> kpm (new KpMatcher(detector, descriptor));
    cv::Mat img = kpm->ReadImage(modelPath, cv::IMREAD_ANYCOLOR );
    kpm->Init(img);

    /*describe & detect frame keypoints*/
    cv::Mat frame = kpm->ReadImage(framePath, cv::IMREAD_ANYCOLOR );
    kpm->DescribeAndDetectFrameKeypoints(frame);

    /*matching*/
    kpm->FindCurrentMatches(xFeatureNorm, ratio1);
    float seconds;
    clock_t t;
    t = clock();
    kpm->ImproveBadMatches(xFeatureNorm, ratio2);
    t = clock() - t;
    seconds = (float)t / CLOCKS_PER_SEC;

    /*~~~~~~~~~~~~~~~~~~~~~ 3D reconstruction (EPFL) ~~~~~~~~~~~~~~~~~~~~~~~~~*/
    unique_ptr<Reconstructor> rec (new Reconstructor());

    rec->init(img);

    std::vector<DMatch> matches = kpm->GetMatches();
    std::vector<KeyPoint> kp1 = kpm->GetModelMatchedKeypoints();
    std::vector<KeyPoint> kp2 = kpm->GetFrameMatchedKeypoints();

    rec->prepareMatches(matches, kp1, kp2);
    rec->deform();
    rec->SetUseTemporal(true);
    rec->SetUsePrevFrameToInit(true);

    if(argv[7] == std::string("1"))
    {
        /*prepare name*/
        std::stringstream ss;
        ss << "PC_" << GetFrameNumber() << "_" << argv[5] << "_" << argv[6] << "_" << argv[3] << "_" << argv[4];

        /*save images*/
        kpm->DrawFoundMatches(img, frame, false, ss.str() + "_all");
        kpm->DrawFoundMatches(img, frame, true, ss.str() + "_imp");
        kpm->DrawKeypointsDisplacement(img, frame, ss.str() + "_disp");

        /*save 3d info*/
        rec->savePointCloud(ss.str());
        rec->drawMesh(frame, rec->resMesh, ss.str() );

        /*save matches*/
        std::ofstream outfile;
        outfile.open("matches.txt", std::ios_base::app);
        outfile << ss.str() << " " << kpm->GetMatches().size() << " " << seconds << endl;
        outfile.close();
    }

    return EXIT_SUCCESS;
}
