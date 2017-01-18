#include "include/affineHelper.h"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

int main()
{
    Ptr<Feature2D> f2d = SIFT::create();
    vector<Mat> descriptors;
    vector<vector<KeyPoint> > keypoints;

    AffineHelper* ah = new AffineHelper(f2d);
    Mat img = ah->readImage("a.png", IMREAD_GRAYSCALE);
    ah->precomputation(img, descriptors, keypoints);

    //single image
    Mat frame = ah->readImage("b.png", IMREAD_GRAYSCALE);
    Mat descriptorsFrame;
    vector<KeyPoint> keypointsFrame;
    f2d->detect(frame, keypointsFrame);
    f2d->compute( frame, keypointsFrame, descriptorsFrame );
    vector<DMatch> matches;
    ah->findCurrentMatches(descriptors[0], descriptorsFrame, matches, NORM_L2, 0.4);
    Mat imgMatches;
    ah->drawFoundMatches(img, keypoints[0], frame, keypointsFrame, matches, imgMatches);
    waitKey(0);
    //


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
