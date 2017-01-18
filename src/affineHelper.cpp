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

void AffineHelper::precomputation(Mat &img, vector<Mat> &allDescriptors, vector<vector<KeyPoint> > &allKeypoints)
{
    cout << "Precomputation started..." << endl;
    Point2f* srcPoints = setupSourcePoints(img);

    cout << "Compute descriptors for input image." << endl;
    describeAndDetect(img, allKeypoints, allDescriptors);

    for(int i = 0; i <= Last; i++)
    {
        Point2f* dstPoints = setupDestinationPoints(srcPoints, (TransformPoints)i, 100);

        Mat dst;
        Mat transform_matrix = getPerspectiveTransform(srcPoints, dstPoints);
        warpPerspective( img, dst, transform_matrix, Size(img.cols, img.rows));

        cout << "Compute descriptors for affine transformation: " << i << endl;
        describeAndDetect(dst, allKeypoints, allDescriptors);

        delete [] dstPoints;
    }
    cout << "Precomputation finished." << endl;
    delete [] srcPoints;
}

void AffineHelper::findCurrentMatches(Mat &descriptors1, Mat &descriptors2, vector<DMatch> &matches, NormTypes norm, const float ratio)
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
    }
}

void AffineHelper::drawFoundMatches(Mat &img1, vector<KeyPoint> &keypoints1, Mat &img2, vector<KeyPoint> &keypoints2, vector<DMatch> &matches, Mat &imgMatches)
{
    namedWindow("matches", 1);
    //drawMatches(img1, keypoints1, img2, keypoints2, matches, imgMatches);

    drawKeypoints(img1, keypoints1, img1);
    drawKeypoints(img2, keypoints2, img2);

    Mat out = Mat(img1.rows*2, img1.cols, img1.type());
    img1.copyTo(out.rowRange(0, img1.rows).colRange(0, img1.cols));
    img2.copyTo(out.rowRange(img1.rows, img1.rows+img2.rows).colRange(0, img2.cols));

    for(int i = 0; i < matches.size(); i++)
    {
        int from = matches[i].queryIdx;
        int to = matches[i].trainIdx;
        cout << from << "   " << to << endl;

        keypoints2[to].pt.y += img1.rows;
        line(out, keypoints1[from].pt, keypoints2[to].pt, Scalar(255, 255, 255), 1);
    }

    imshow("matches", out);
}


//
//private section
//
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

void AffineHelper::describeAndDetect(Mat &img, vector<vector<KeyPoint> > &keypoints, vector<Mat> &descriptors)
{
    vector<KeyPoint> keypointsPre;
    alghoritm->detect(img, keypointsPre);
    keypoints.push_back(keypointsPre);

    Mat desc;
    alghoritm->compute( img, keypointsPre, desc );
    descriptors.push_back(desc);

    keypointsPre.clear();
    desc.release();
}
