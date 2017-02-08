#include "include/affineHelper.h"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

int main()
{
    Ptr<Feature2D> detector = SURF::create();
    Ptr<Feature2D> descriptor = SIFT::create();
    vector<Mat> descriptors;
    vector<vector<KeyPoint> > keypoints;

    AffineHelper* ah = new AffineHelper(detector, descriptor);
    Mat img = ah->readImage("images/a.png", IMREAD_GRAYSCALE );
    Mat roi = img(Rect(Point2f(180, 60), Point2f(600, 390)));
    ah->precomputation(roi , descriptors, keypoints, true, true);

    //single image
    Mat frameMatRect = ah->readImage("images/b.png", IMREAD_GRAYSCALE );
    resize(frameMatRect, frameMatRect, Size(600, 400));
    Mat frame = frameMatRect(Rect(Point2f(180, 60), Point2f(600, 390)));
    Mat descriptorsFrame;
    vector<KeyPoint> keypointsFrame;
    detector->detect(frame, keypointsFrame);
    descriptor->compute( frame, keypointsFrame, descriptorsFrame );
    vector<DMatch> matches, badMatches;
    ah->findCurrentMatches(descriptors[0], descriptorsFrame, matches, badMatches, NORM_L2, 0.9);

    int goodMatchesIdx = matches.size();

    Mat imgMatches1;
    ah->drawFoundMatches(img , keypoints[0], frame, keypointsFrame, matches, imgMatches1, "precompute step");
    cout << "Number of matches before improvement: " << matches.size() << endl;

    ah->improveBadMatches(descriptors, descriptorsFrame, matches, badMatches, NORM_L2, true);
    cout << "Number of matches after improvement: " << matches.size() << endl;

    //goodMatches
    Mat imgMatches;
    vector<DMatch> goodMatches;
    for(int j = goodMatchesIdx; j < matches.size(); j++)
    {
        goodMatches.push_back(matches[j]);
    }
    ah->drawFoundMatches(img, keypoints[0], frame, keypointsFrame, goodMatches, imgMatches, "improved matching");
    //ah->drawFoundMatches(img, keypoints[0], frame, keypointsFrame, matches, imgMatches, "improved matching");


//    cout << "Open default camera..." << endl;
//    VideoCapture cap;
//    if(!cap.open(0))
//    {
//        cerr << "Cannot open default camera." << endl;
//        return 0;
//    }

//    cout << "Capturing." << endl;
//    for(;;)
//    {
//        Mat descriptorsFrame;
//        vector<KeyPoint> keypointsFrame;

//        Mat frame;
//        cap >> frame;
//        if( frame.empty() )
//        {
//            break;
//        }

//        //detect keypoints on frame
//        f2d->detect(frame, keypointsFrame);

//        //describe keypoints
//        f2d->compute( frame, keypointsFrame, descriptorsFrame );

//        //match keypoints
//        vector<DMatch> matches;
//        ah->findCurrentMatches(descriptors[0], descriptorsFrame, matches, NORM_HAMMING, 0.8);

//        //cout << keypoints[0].size() << endl;

//        // draw matches
//        if( !keypoints[0].empty() && !keypointsFrame.empty() )
//        {
//            Mat imgMatches;
//            ah->drawFoundMatches(img, keypoints[0], frame, keypointsFrame, matches, imgMatches);
//        }

//        if( waitKey(1) == 27 ) break;
//    }

    free(ah);
    return 0;
}
