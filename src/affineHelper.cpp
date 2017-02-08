#include "../include/affineHelper.h"

//
// public section
//
Mat AffineHelper::readImage(const string &name, ImreadModes mode)
{
    Mat img = imread(name, mode);
    if ( img.empty() )
    {
        cerr << "Error while loading an image " << name << endl;
        abort();
    }
    cout << "Image " << name << " properly loaded" << endl;
    return img;
}


void AffineHelper::precomputation(Mat &img, vector<Mat> &allDescriptors, vector<vector<KeyPoint> > &allKeypoints, bool isRoi, bool isChessboardPresent)
{
    //resize(img, img, Size(600, 400));
    if(isChessboardPresent)
    {
        cout << "Precomputation started..." << endl;
        vector<Mat> transformationMatrixes;
        getTransformationMatrixes("images/chessboard.txt", transformationMatrixes);

        cout << "Compute descriptors for input image." << endl;
        describeAndDetectInitKeypoints(img, allKeypoints, allDescriptors);
        vector<KeyPoint> initialVector = allKeypoints[0];

        for(int i = 0; i < transformationMatrixes.size(); i++)
        {
            Mat dst = Mat::zeros( img.rows, img.cols, img.type() );
            warpPerspective( img, dst, transformationMatrixes[i], Size(img.cols, img.rows));

            cout << "Compute descriptors for affine transformation: " << i << endl;
            describeAndDetectTransformedKeypoints(dst, transformationMatrixes[i], initialVector, allKeypoints, allDescriptors);
        }
        cout << "Precomputation finished." << endl;
    }
    else
    {
        cout << "Precomputation started..." << endl;
        Point2f* srcPoints = setupSourcePoints(img);

        cout << "Compute descriptors for input image." << endl;
        describeAndDetectInitKeypoints(img, allKeypoints, allDescriptors);
        vector<KeyPoint> initialVector = allKeypoints[0];

        for(int i = 0; i < Last; i++)
        {
            Mat dst = Mat::zeros( img.rows, img.cols, img.type() );
            Point2f* dstPoints = setupDestinationPoints(srcPoints, (TransformPoints)i, 50);


            Mat transformMatrix = getPerspectiveTransform (srcPoints, dstPoints);
            warpPerspective( img, dst, transformMatrix, Size(img.cols, img.rows));

            cout << "Compute descriptors for affine transformation: " << i << endl;
            describeAndDetectTransformedKeypoints(dst, transformMatrix, initialVector, allKeypoints, allDescriptors);

            delete [] dstPoints;
        }
        cout << "Precomputation finished." << endl;
        delete [] srcPoints;
    }
    if(isRoi)
    {
        for(int i = 0; i < allKeypoints.size(); i++)
        {
            for(int j = 0; j < allKeypoints[i].size(); j++)
            {
                allKeypoints[i][j].pt.x += 180;
                allKeypoints[i][j].pt.y += 60;
            }
        }
    }
}

void AffineHelper::findCurrentMatches(Mat &descriptors1, Mat &descriptors2, vector<DMatch> &matches, vector<DMatch> &badMatches, NormTypes norm, const float ratio)
{
    vector<vector<DMatch> > potentialMatches;
    BFMatcher matcher(norm);

    matcher.knnMatch(descriptors1, descriptors2, potentialMatches, 2);

    for (int i = 0; i < potentialMatches.size(); ++i)
    {
        if (potentialMatches[i][0].distance < ratio * potentialMatches[i][1].distance)
        {
            matches.push_back(potentialMatches[i][0]);
        }
        else
        {
            badMatches.push_back(potentialMatches[i][0]);
        }
    }
}

Mat AffineHelper::getOnePointDescriptors(int index, vector<Mat> &descriptors)
{
    Mat temp;

    for(int i=0;i<descriptors.size();i++)
    {
        if(index < descriptors[i].rows)
        {
            temp.push_back(descriptors[i].row(index));
        }
    }
    return temp;
}

void AffineHelper::improveBadMatches(vector<Mat> &descriptors1, Mat &descriptors2, vector<DMatch> &goodMatches, vector<DMatch> &badMatches, NormTypes norm, bool isFlann)
{
    DescriptorMatcher* matcher;
    if(isFlann)
    {
        matcher = new FlannBasedMatcher();
    }
    else
    {
        matcher = new BFMatcher(norm);
    }

    vector<vector<DMatch> > potentialMatches;

    cout << "Improve the quality of matching using affine transformations..." << endl;
    for(int i = 0; i < badMatches.size(); i++)
    {
        int originalFotoIdx = badMatches[i].queryIdx;   ///
        int comparedFotoIdx = badMatches[i].trainIdx;
        int originalDistance = badMatches[i].distance;
        Mat onePointDesc = getOnePointDescriptors(originalFotoIdx, descriptors1);

        //cout << "Bad Matches: " << endl << "originalFotoIdx: " << originalFotoIdx << "comparedFotoIdx: " << originalDistance << "originalDistance: " << originalDistance << "onePointDesc.size(): " << onePointDesc.size() << endl;
        if(!onePointDesc.empty())
        {
            Mat desc2;
            desc2.push_back(descriptors2.row(comparedFotoIdx));
            matcher->knnMatch( desc2, onePointDesc, potentialMatches, 1 );
            if(!potentialMatches.empty() && potentialMatches[0][0].distance < 0.9 * originalDistance)
            {
                cout << "Improved - keypoint nr " << badMatches[i].trainIdx << ", new distance: " << potentialMatches[0][0].distance << ", old: " << originalDistance << endl;
                badMatches[i].distance = potentialMatches[0][0].distance;
                goodMatches.push_back(badMatches[i]);
            }
            potentialMatches.clear();
            onePointDesc.release();
            desc2.release();
        }
    }
}

void AffineHelper::drawFoundMatches(Mat &img1, vector<KeyPoint> &keypoints1, Mat &img2, vector<KeyPoint> &keypoints2, vector<DMatch> &matches, Mat &imgMatches, const string windowName)
{
    Mat temp1, temp2;
    img1.copyTo(temp1);
    img2.copyTo(temp2);
    namedWindow("matches", 1);
    drawMatches(temp1, keypoints1, temp2, keypoints2, matches, imgMatches);
    imshow(windowName, imgMatches);
    waitKey(0);
    //    drawKeypoints(img1, keypoints1, temp1);
    //    drawKeypoints(img2, keypoints2, temp2);

    //    Mat out = Mat(img1.rows*2, img1.cols, img1.type());
    //    img1.copyTo(out.rowRange(0, img1.rows).colRange(0, img1.cols));
    //    img2.copyTo(out.rowRange(img1.rows, img1.rows+img2.rows).colRange(0, img2.cols));

    //    for(int i = 0; i < matches.size(); i++)
    //    {
    //        int from = matches[i].trainIdx;
    //        int to = matches[i].queryIdx;

    //        Point2f kpFrom, kpTo;
    //        kpFrom.x = keypoints1[from].pt.x;
    //        kpFrom.y = keypoints1[from].pt.y;
    //        kpTo.x = keypoints2[to].pt.x;
    //        kpTo.y = img1.rows + keypoints2[to].pt.y;
    //        line(out, kpFrom, kpTo, Scalar(255, 255, 255), 1);
    //    }

    //    imshow("matches", out);
    //    waitKey(0);
}




//
//private section
//
void AffineHelper::setUpRotationMatrix(Mat in, Mat &rotationMatrix, float angle, float u, float v, float w)
{
    float L = (u*u + v * v + w * w);
    float u2 = u * u;
    float v2 = v * v;
    float w2 = w * w;

    rotationMatrix.at<float>(0, 0) = (u2 + (v2 + w2) * cos(angle)) / L;
    rotationMatrix.at<float>(0, 1) = (u * v * (1 - cos(angle)) - w * sqrt(L) * sin(angle)) / L;
    rotationMatrix.at<float>(0, 2) = (u * w * (1 - cos(angle)) + v * sqrt(L) * sin(angle)) / L;
    //    rotationMatrix.at<float>(0, 3)  = 0.0;

    //std:: cout << rotationMatrix[0][0] << std::endl;

    rotationMatrix.at<float>(1, 0) = (u * v * (1 - cos(angle)) + w * sqrt(L) * sin(angle)) / L;
    rotationMatrix.at<float>(1, 1) = (v2 + (u2 + w2) * cos(angle)) / L;
    rotationMatrix.at<float>(1, 2) = (v * w * (1 - cos(angle)) - u * sqrt(L) * sin(angle)) / L;
    //    rotationMatrix.at<float>(1, 3) = 0.0;

    rotationMatrix.at<float>(2, 0) = (u * w * (1 - cos(angle)) - v * sqrt(L) * sin(angle)) / L;
    rotationMatrix.at<float>(2, 1)= (v * w * (1 - cos(angle)) + u * sqrt(L) * sin(angle)) / L;
    rotationMatrix.at<float>(2, 2) = (w2 + (u2 + v2) * cos(angle)) / L;
    //    rotationMatrix.at<float>(2, 3) = 0.0;

    //    rotationMatrix.at<float>(3, 0) = 0.0;
    //    rotationMatrix.at<float>(3, 1)= 0.0;
    //    rotationMatrix.at<float>(3, 2) = 0.0;
    //    rotationMatrix.at<float>(3, 3) = 1.0;

    //normalize(rotationMatrix, rotationMatrix, 0, 1, NORM_MINMAX, CV_32FC1);
}

Point2f* AffineHelper::setupSourcePoints(Mat& img)
{
    Point2f* points = new Point2f[NPOINTS];
    points[0] = Point2f(0.0, 0.0);
    points[1] = Point2f(img.cols, 0.0);
    points[2] = Point2f(img.cols, img.rows);
    points[3] = Point2f(0.0, img.rows);
    return points;
}

Point2f* AffineHelper::setupDestinationPoints(Point2f *pointsSrc, TransformPoints value, int factor)
{
    Point2f *pointsDst = cpyArray(pointsSrc, NPOINTS);
    switch(value)
    {
    case Top:
        pointsDst[0] = Point2f(pointsDst[0].x + factor, pointsDst[0].y);
        pointsDst[1] = Point2f(pointsDst[1].x - factor, pointsDst[1].y);
        break;
    case TopRight:
        pointsDst[0] = Point2f(pointsDst[0].x + factor, pointsDst[0].y);
        pointsDst[2] = Point2f(pointsDst[2].x - factor, pointsDst[2].y);
        break;
    case Right:
        pointsDst[1] = Point2f(pointsDst[1].x, pointsDst[1].y + factor);
        pointsDst[2] = Point2f(pointsDst[2].x, pointsDst[2].y - factor);
        break;
    case LowRight:
        pointsDst[1] = Point2f(pointsDst[1].x - factor, pointsDst[1].y);
        pointsDst[3] = Point2f(pointsDst[3].x + factor, pointsDst[3].y);
        break;
    case Low:
        pointsDst[2] = Point2f(pointsDst[2].x - factor, pointsDst[2].y);
        pointsDst[3] = Point2f(pointsDst[3].x + factor, pointsDst[3].y);
        break;
    case LowLeft:
        pointsDst[0] = Point2f(pointsDst[0].x + factor, pointsDst[0].y);
        pointsDst[2] = Point2f(pointsDst[2].x - factor, pointsDst[2].y);
        break;
    case Left:
        pointsDst[0] = Point2f(pointsDst[0].x, pointsDst[0].y + factor);
        pointsDst[3] = Point2f(pointsDst[3].x, pointsDst[3].y - factor);
        break;
    case TopLeft:
        pointsDst[1] = Point2f(pointsDst[1].x - factor, pointsDst[1].y);
        pointsDst[3] = Point2f(pointsDst[3].x + factor, pointsDst[3].y);
        break;
    }
    return pointsDst;
}

Point2f* AffineHelper::cpyArray(Point2f* in, int size)
{
    Point2f* out = new Point2f[NPOINTS];
    for(int i = 0; i<size; i++)
    {
        out[i] = in[i];
    }
    return out;
}

void AffineHelper::describeAndDetectInitKeypoints(Mat &img, vector<vector<KeyPoint> > &keypoints, vector<Mat> &descriptors)
{
    vector<KeyPoint> keypointsPre;
    detector->detect(img, keypointsPre);
    keypoints.push_back(keypointsPre);

    Mat desc;
    descriptor->compute( img, keypointsPre, desc );

//    computeDescriptorsInProperOrder(keypointsPre, desc, img);
    descriptors.push_back(desc);


    keypointsPre.clear();
    desc.release();
}

void AffineHelper::computeDescriptorsInProperOrder(vector<KeyPoint> &kp, Mat &d, Mat &img)
{
    for(int i = 0; i < kp.size(); i++)
    {
        vector<KeyPoint> tempKp;
        Mat tempDesc;
        tempKp.push_back(kp[i]);
        descriptor->compute( img, tempKp, tempDesc );
        d.push_back(tempDesc);
        tempDesc.release();
        tempKp.clear();
    }
}

void AffineHelper::describeAndDetectTransformedKeypoints(Mat &transformedImg, Mat &transformMatrix, vector<KeyPoint> &initialVector, vector<vector<KeyPoint> > &keypoints, vector<Mat> &descriptors)
{
    vector<KeyPoint> keypointsPre = initialVector;
    vector<Point2f> kp;
    for(int i = 0; i < initialVector.size(); i++)
    {
        kp.push_back(initialVector[i].pt);
    }

    perspectiveTransform(kp, kp, transformMatrix);

    for(int i = 0; i < keypointsPre.size(); i++)
    {
        keypointsPre[i].pt = kp[i];
    }

    keypoints.push_back(keypointsPre);

    Mat desc;
    descriptor->compute( transformedImg, keypointsPre, desc );
    descriptors.push_back(desc);

    Mat temp;
    drawKeypoints(transformedImg, keypointsPre, temp );
    imshow("aaa", temp);
    waitKey(0);

    keypointsPre.clear();
    kp.clear();
    desc.release();
}
void AffineHelper::findCorners( Mat &image, vector<Point2f> &corners )
{
    Size pattern = Size(5,4);
    bool isFound = findChessboardCorners(image, pattern, corners);
    if(!isFound)
    {
        cerr << "Error occured while procssing chessboard data." << endl;
        abort();
    }
}

void AffineHelper::getTransformationMatrixes(const string &path, vector<Mat> &matrixes)
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
            Mat nextImage = readImage(line, IMREAD_GRAYSCALE);
            resize(nextImage, nextImage, Size(600, 400));
            findCorners(nextImage, nextCorners);
            Mat homographyMatrix = findHomography(initialCorners, nextCorners, RANSAC);
            matrixes.push_back(homographyMatrix);
        }
        else
        {
            Mat initialImage = readImage(line, IMREAD_GRAYSCALE);
            findCorners(initialImage, initialCorners);
        }
        i++;
    }

    infile.close();
}
