#include "include/affineHelper.h"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

int main()
{
    Ptr<Feature2D> detector = SIFT::create();
    Ptr<Feature2D> descriptor = SIFT::create();
    vector<Mat> descriptors;
    vector<vector<KeyPoint> > keypoints;

    AffineHelper* ah = new AffineHelper(detector, descriptor);
    Mat img = ah->readImage("images/a.png", IMREAD_GRAYSCALE );
    Mat roi = img(Rect(Point2f(180, 80), Point2f(600, 410)));
    ah->precomputation(roi, descriptors, keypoints, true, true);


    //read sequence
    VideoCapture capture = cv::VideoCapture( "output.avi" );

    if(!capture.isOpened())  // check if we succeeded
        return -1;

    int idx=1;
    while(true)
    {
        Mat frame;
        capture >> frame;

        if(frame.empty())
        {
            break;
        }

        cvtColor(frame, frame, CV_BGR2GRAY);

        Mat descriptorsFrame;
        vector<KeyPoint> keypointsFrame;
        detector->detect(frame, keypointsFrame);
        descriptor->compute( frame, keypointsFrame, descriptorsFrame );

        vector<DMatch> matches, badMatches;
        ah->findCurrentMatches(descriptors[0], descriptorsFrame, matches, badMatches, NORM_L2, 0.8);
        ah->improveBadMatches(descriptors, descriptorsFrame, matches, badMatches, NORM_L2, true);

        //print to file
        ofstream matchFrame;
        stringstream ss;
        ss << "output_files/match_" << fixed << setprecision(3) << idx << ".txt";
        string matchFileName = ss.str();
        ss.str("");
        matchFrame.open (matchFileName.c_str(), ofstream::out | ofstream::trunc);
        for(int j = 0; j < matches.size(); j++)
        {
            matchFrame << matches[j].queryIdx << " " << matches[j].trainIdx << endl;
        }
        matchFrame.close();

        ofstream kpFrame;
        ss << "output_files/kp_" << fixed << setprecision(3) << idx << ".txt";
        string kpFileName = ss.str();
        ss.str("");
        kpFrame.open (kpFileName.c_str(), ofstream::out | ofstream::trunc);
        for(int j = 0; j < keypointsFrame.size(); j++)
        {
            kpFrame << keypointsFrame[j].pt.x << " " << keypointsFrame[j].pt.y << endl;
        }
        kpFrame.close();

        //release vectors
        matches.clear();
        keypointsFrame.clear();
        idx++;
    }

//    //Mat frameMatRect = ah->readImage("images/frame_128.png", IMREAD_GRAYSCALE );
//    Mat frame = frameMatRect(Rect(Point2f(180, 80), Point2f(600, 410)));
//    Mat descriptorsFrame;
//    vector<KeyPoint> keypointsFrame;
//    detector->detect(frame, keypointsFrame);
//    descriptor->compute( frame, keypointsFrame, descriptorsFrame );

//    vector<DMatch> matches, badMatches;
//    ah->findCurrentMatches(descriptors[0], descriptorsFrame, matches, badMatches, NORM_L2, 0.8);
//    int goodMatchesIdx = matches.size();
//    Mat imgMatches1;
//    ah->drawFoundMatches(img , keypoints[0], frame, keypointsFrame, matches, imgMatches1, "precompute step");
//    cout << "Number of matches before improvement: " << matches.size() << endl;

//    ah->improveBadMatches(descriptors, descriptorsFrame, matches, badMatches, NORM_L2, true);
//    cout << "Number of matches after improvement: " << matches.size() << endl;

//    //goodMatches
//    Mat imgMatches;
//    vector<DMatch> goodMatches;
//    for(int j = goodMatchesIdx; j < matches.size(); j++)
//    {
//        goodMatches.push_back(matches[j]);
//    }
//    ah->drawFoundMatches(img, keypoints[0], frame, keypointsFrame, goodMatches, imgMatches, "improved matching");


    /*kp init [a b]*/
//    ofstream kpFile1;
//    string kpFileName1 = "keypoints1.txt";
//    kpFile1.open (kpFileName1.c_str(), ofstream::out | ofstream::trunc);
//    for(int j = 0; j < keypoints[0].size(); j++)
//    {
//        kpFile1 << keypoints[0][j].pt.x << " " << keypoints[0][j].pt.y << endl;
//    }
//    kpFile1.close();

//    /*kp frame [a b]*/
//    ofstream kpFile;
//    string kpFileName = "keypoints2.txt";
//    kpFile.open (kpFileName.c_str(), ofstream::out | ofstream::trunc);
//    for(int j = 0; j < keypointsFrame.size(); j++)
//    {
//        kpFile << keypointsFrame[j].pt.x + 180 << " " << keypointsFrame[j].pt.y + 80 << endl;
//    }
//    kpFile.close();

//    /*matches [a b c d] and indexes [a b]*/
//    ofstream myfile;
//    ofstream myfile1;
//    string filename = "matches.txt";
//    string filename1 = "indexes.txt";
//    myfile.open (filename.c_str(), ofstream::out | ofstream::trunc);
//    myfile1.open (filename1.c_str(), ofstream::out | ofstream::trunc);
//    for(int j = 0; j < matches.size(); j++)
//    {
//        int idx1 = matches[j].queryIdx; //original photo desc idx
//        int idx2 = matches[j].trainIdx; //compared photo desc idx
//        myfile << keypoints[0][idx1].pt.x << " " << keypoints[0][idx1].pt.y << " " << keypointsFrame[idx2].pt.x << " " << keypointsFrame[idx2].pt.y << endl;
//        myfile1 << idx1 << " " << idx2 << endl;
//    }
//    myfile.close();
//    myfile1.close();

    free(ah);
    return 0;
}
