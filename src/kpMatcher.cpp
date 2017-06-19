#include "../include/kpMatcher.h"
#include <opencv2/opencv.hpp>

//
// public section
//
Mat KpMatcher::readImage(const string &name, ImreadModes mode)
{
    Mat img = imread(name, mode);
    if ( img.empty() )
    {
        cerr << "Error while loading an image " << name << endl;
        abort();
    }
    resize(img, img, Size(640, 480));
    cout << "Image " << name << " properly loaded" << endl;
    return img;
}


void KpMatcher::init(const Mat &img)
{
    cout << "Compute descriptors for input image." << endl;
    describeAndDetectInitKeypoints(img);

    if(this->isSimulatedLocalDeform)
    {
        cout << "Initialization started..." << endl;
        getTransformationMatrixes("images/chessboard.txt");

        for(size_t i = 0; i < this->transformationMatrixes.size(); i++)
        {
            Mat dst = Mat::zeros( img.rows, img.cols, img.type() );
            warpPerspective( img, dst, this->transformationMatrixes[i], Size(img.cols, img.rows));

            cout << "Compute descriptors for affine transformation: " << i << endl;
            describeAndDetectTransformedKeypoints(dst, this->transformationMatrixes[i]);
        }
    }
    cout << "Initialization finished." << endl;
}

void KpMatcher::describeAndDetectFrameKeypoints(const Mat& frame)
{
    this->detector->detect(frame, keypointsFrame);
    this->descriptor->compute( frame, keypointsFrame, descriptorsFrame );
}

void KpMatcher::findCurrentMatches(NormTypes norm, const float &ratio)
{
    DescriptorMatcher* matcher = new BFMatcher(norm);

    vector<vector<DMatch> > potentialMatches;
    matcher->knnMatch(this->descriptors[0], this->descriptorsFrame, potentialMatches, 2);

    cout << potentialMatches.size() << endl;

    for (size_t i = 0; i < potentialMatches.size(); ++i)
    {
        if (potentialMatches[i][0].distance < ratio * potentialMatches[i][1].distance)
        {
            this->goodMatches.push_back(potentialMatches[i][0]);
        }
        else
        {
            this->badMatches.push_back(potentialMatches[i][0]);
        }
    }
}

Mat KpMatcher::getOnePointDescriptors(int index)
{
    Mat temp;

    for(size_t i=1;i<this->descriptors.size();i++)
    {
        if(index < this->descriptors[i].rows)
        {
            temp.push_back(this->descriptors[i].row(index));
        }
    }
    return temp;
}

void KpMatcher::improveBadMatches(NormTypes norm, const float &ratio)
{
    if(this->isSimulatedLocalDeform)
    {
        cout << "RATIO: " << ratio << endl;
        DescriptorMatcher* matcher = new BFMatcher(norm);

        vector<vector<DMatch> > potentialMatches;

        cout << "Improve the quality of matching using affine transformations..." << endl;
        for(size_t i = 0; i < this->badMatches.size(); i++)
        {
            int originalFotoIdx = this->badMatches[i].queryIdx;
            int comparedFotoIdx = this->badMatches[i].trainIdx;
            float originalDistance = this->badMatches[i].distance;
            Mat onePointDesc = getOnePointDescriptors(originalFotoIdx);

            //cout << "Bad Matches: " << endl << "originalFotoIdx: " << originalFotoIdx << "comparedFotoIdx: " << originalDistance << "originalDistance: " << originalDistance << "onePointDesc.size(): " << onePointDesc.size() << endl;
            if(!onePointDesc.empty())
            {
                Mat desc2;
                KeyPoint to = this->keypointsFrame[comparedFotoIdx];
                KeyPoint from = this->keypoints[0][originalFotoIdx];
                double dist = sqrt((from.pt.x - to.pt.x) * (from.pt.x - to.pt.x) + (from.pt.y - to.pt.y) * (from.pt.y - to.pt.y));
                desc2.push_back(this->descriptorsFrame.row(comparedFotoIdx));
                matcher->knnMatch( desc2, onePointDesc, potentialMatches, 2 );
                if(dist < 30 && !potentialMatches.empty() && potentialMatches[0][0].distance < ratio * potentialMatches[0][1].distance && potentialMatches[0][0].distance < originalDistance )
                {
                    //cout << "Improved - keypoint nr " << this->badMatches[i].queryIdx << ", new distance: " << potentialMatches[0][0].distance << ", old: " << originalDistance << endl;
                    badMatches[i].distance = potentialMatches[0][0].distance;
                    this->goodMatches.push_back(this->badMatches[i]);
                    this->improvementMatches.push_back(this->badMatches[i]);
                }
                potentialMatches.clear();
                onePointDesc.release();
                desc2.release();
            }
        }
    }
}

void KpMatcher::drawKeypointsDisplacement(const Mat& img1, const Mat& img2, string fileName)
{
    fileName = fileName + ".png";
    Mat temp1, temp2;
    img1.copyTo(temp1);
    img2.copyTo(temp2);
    if(!this->improvementMatches.empty() )
    {
        for(size_t i = 0; i < this->improvementMatches.size(); ++i)
        {
            KeyPoint to = this->keypointsFrame[this->improvementMatches[i].trainIdx];
            KeyPoint from = this->keypoints[0][this->improvementMatches[i].queryIdx];
            line(temp2, from.pt, to.pt, Scalar(0, 0, 255), 2, 8, 0);
        }
        cout << "Size of improved matches: " <<  this->improvementMatches.size() << endl;
    }
    else
    {
        cerr << "Cannot show matches. Check if you performed improvement matches process." << endl;
        return;
    }
    imwrite( fileName, temp2 );
}

void KpMatcher::drawFoundMatches(const Mat& img1, const Mat& img2, bool drawOnlyImproved, string fileName)
{
    fileName = fileName + ".png";
    Mat temp1, temp2, imgMatches;
    img1.copyTo(temp1);
    img2.copyTo(temp2);
    if(drawOnlyImproved && !this->improvementMatches.empty() )
    {
        drawMatches(temp1, this->keypoints[0], temp2, this->keypointsFrame, this->improvementMatches, imgMatches);
        cout << "Size of improved matches: " <<  this->improvementMatches.size() << endl;
    }
    else if(!drawOnlyImproved)
    {
        drawMatches(temp1, this->keypoints[0], temp2, this->keypointsFrame, this->goodMatches, imgMatches);
        cout << "Size of all keypoints: " <<  this->keypoints[0].size() << endl;
        cout << "Size of all matches: " <<  this->goodMatches.size() << endl;
    }
    else
    {
        cerr << "Cannot show matches. Check if you performed improvement matches process." << endl;
        return;
    }
    imwrite( fileName, imgMatches );
}

vector<DMatch> KpMatcher::getMatches()
{
    return this->goodMatches;
}

vector<KeyPoint> KpMatcher::getModelMatchedKeypoints()
{
    return this->keypoints[0];
}

vector<KeyPoint> KpMatcher::getFrameMatchedKeypoints()
{
    return this->keypointsFrame;
}



//
//private section
//

bool KpMatcher::isInParalellgoram(KeyPoint& kp)
{
    //couter-clockwise
    for(int i = parallellogram.size(); i > 0; i--)
    {
        float A = -(parallellogram[i-1].y - parallellogram[i].y);
        float B = parallellogram[i-1].x - parallellogram[i].x;
        float C = -(A * parallellogram[i].x + B * parallellogram[i].y);
        float D = A * kp.pt.x + B * kp.pt.y + C;
        if(D > 0)
        {
            return false;
        }
    }
    return true;
}

void KpMatcher::describeAndDetectInitKeypoints(const Mat &img)
{
    vector<KeyPoint> keypointsPre;
    detector->detect(img, keypointsPre);
    for(size_t i = 0; i < keypointsPre.size(); i++)
    {
        if(!isInParalellgoram(keypointsPre[i]))
        {
            keypointsPre.erase(keypointsPre.begin() + i);
            i--;
        }
    }
    this->keypoints.push_back(keypointsPre);

    Mat desc;
    this->descriptor->compute( img, keypointsPre, desc );
    this->descriptors.push_back(desc);

    keypointsPre.clear();
    desc.release();
}

void KpMatcher::describeAndDetectTransformedKeypoints(const Mat& transformedImg, const Mat& transformMatrix)
{
    vector<KeyPoint> initialVector(this->keypoints[0]);
    vector<KeyPoint> keypointsPre(initialVector);
    vector<Point2f> kp;
    for(size_t i = 0; i < initialVector.size(); i++)
    {
        kp.push_back(initialVector[i].pt);
    }
    perspectiveTransform(kp, kp, transformMatrix);
    for(size_t i = 0; i < keypointsPre.size(); i++)
    {
        keypointsPre[i].pt = kp[i];
    }
    this->keypoints.push_back(keypointsPre);

//    drawKeypoints(transformedImg, keypointsPre, transformedImg);
//    imshow("keypointsPre", transformedImg);
//    waitKey(0);

    Mat desc;
    this->descriptor->compute( transformedImg, keypointsPre, desc );
    this->descriptors.push_back(desc);
}

void KpMatcher::findCorners(const Mat &image, vector<Point2f> &corners )
{
    //TODO read chessboard size from file
    Size pattern = Size(9,6);
    bool isFound = findChessboardCorners(image, pattern, corners);
    if(!isFound)
    {
        cerr << "Error occured while procssing chessboard data." << endl;
        abort();
    }
}

void KpMatcher::getTransformationMatrixes(const string &path)
{
    ifstream infile(path.c_str());
    vector<Point2f> initialCorners;
    int i = 0;
    string line;
    while (getline(infile, line))
    {
        if(i > 0)
        {
            vector<Point2f> nextCorners;
            Mat nextImage = readImage(line, IMREAD_ANYCOLOR);
            findCorners(nextImage, nextCorners);
            Mat homographyMatrix = findHomography(initialCorners, nextCorners, RANSAC);
            this->transformationMatrixes.push_back(homographyMatrix);
        }
        else
        {
            Mat initialImage = readImage(line, IMREAD_ANYCOLOR);
            findCorners(initialImage, initialCorners);

//            namedWindow("aaa", WINDOW_NORMAL);
//            imshow("aaa", initialImage);
//            waitKey(0);
        }
        i++;
    }
    infile.close();
}
