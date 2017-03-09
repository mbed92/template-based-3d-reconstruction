#include "../include/kpMatcher.h"

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
    cout << "Image " << name << " properly loaded" << endl;
    return img;
}


void KpMatcher::init(Mat &img, bool isChessboardPresent)
{
    //if chessboard images are used to perform perspective transformations
    if(isChessboardPresent)
    {
        cout << "Initialization started..." << endl;
        getTransformationMatrixes("images/chessboard.txt");

        cout << "Compute descriptors for input image." << endl;
        describeAndDetectInitKeypoints(img);

        for(int i = 0; i < this->transformationMatrixes.size(); i++)
        {
            Mat dst = Mat::zeros( img.rows, img.cols, img.type() );
            warpPerspective( img, dst, this->transformationMatrixes[i], Size(img.cols, img.rows));

            cout << "Compute descriptors for affine transformation: " << i << endl;
            describeAndDetectTransformedKeypoints(dst, this->transformationMatrixes[i]);
        }
        cout << "Initialization finished." << endl;
    }
    else
    {
        cout << "Initialization started..." << endl;
        Point2f* srcPoints = setupSourcePoints(img);

        cout << "Compute descriptors for input image." << endl;
        describeAndDetectInitKeypoints(img);

        for(int i = 0; i < Last; i++)
        {
            Mat dst = Mat::zeros( img.rows, img.cols, img.type() );
            Point2f* dstPoints = setupDestinationPoints(srcPoints, (TransformPoints)i, 50);


            Mat transformMatrix = getPerspectiveTransform (srcPoints, dstPoints);
            warpPerspective( img, dst, transformMatrix, Size(img.cols, img.rows));

            cout << "Compute descriptors for affine transformation: " << i << endl;
            describeAndDetectTransformedKeypoints(dst, transformMatrix);

            delete [] dstPoints;
        }
        cout << "Initialization finished." << endl;
        delete [] srcPoints;
    }
}

void KpMatcher::describeAndDetectFrameKeypoints(Mat &frame)
{
    this->detector->detect(frame, keypointsFrame);
    this->descriptor->compute( frame, keypointsFrame, descriptorsFrame );
}

void KpMatcher::findCurrentMatches(NormTypes norm, bool isFlann, const float ratio)
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
    matcher->knnMatch(this->descriptors[0], this->descriptorsFrame, potentialMatches, 2);

    for (int i = 0; i < potentialMatches.size(); ++i)
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

    for(int i=1;i<this->descriptors.size();i++)
    {
        if(index < this->descriptors[i].rows)
        {
            temp.push_back(this->descriptors[i].row(index));
        }
    }
    return temp;
}

void KpMatcher::improveBadMatches(NormTypes norm, bool isFlann, const float ratio)
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
    for(int i = 0; i < this->badMatches.size(); i++)
    {
        int originalFotoIdx = this->badMatches[i].queryIdx;
        int comparedFotoIdx = this->badMatches[i].trainIdx;
        int originalDistance = this->badMatches[i].distance;
        Mat onePointDesc = getOnePointDescriptors(originalFotoIdx);

        //cout << "Bad Matches: " << endl << "originalFotoIdx: " << originalFotoIdx << "comparedFotoIdx: " << originalDistance << "originalDistance: " << originalDistance << "onePointDesc.size(): " << onePointDesc.size() << endl;
        if(!onePointDesc.empty())
        {
            Mat desc2;
            desc2.push_back(this->descriptorsFrame.row(comparedFotoIdx));
            matcher->knnMatch( desc2, onePointDesc, potentialMatches, 1 );
            if(!potentialMatches.empty() && potentialMatches[0][0].distance < ratio * originalDistance)
            {
                cout << "Improved - keypoint nr " << this->badMatches[i].queryIdx << ", new distance: " << potentialMatches[0][0].distance << ", old: " << originalDistance << endl;
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

void KpMatcher::drawFoundMatches(Mat &img1, Mat &img2, const string &windowName, bool drawOnlyImproved)
{
    Mat temp1, temp2, imgMatches;
    img1.copyTo(temp1);
    img2.copyTo(temp2);
    if(drawOnlyImproved && !this->improvementMatches.empty() )
    {
        drawMatches(temp1, this->keypoints[0], temp2, this->keypointsFrame, this->improvementMatches, imgMatches);
    }
    else if(!drawOnlyImproved)
    {
        drawMatches(temp1, this->keypoints[0], temp2, this->keypointsFrame, this->goodMatches, imgMatches);
    }
    else
    {
        cerr << "Cannot show matches. Check if you performed improvement matches process." << endl;
        return;
    }
    imshow(windowName, imgMatches);
    waitKey(0);
}

//void KpMatcher::drawAllMatches(Mat &img1, Mat &img2, const string &windowName)
//{
//    Mat temp1, temp2, imgMatches;
//    img1.copyTo(temp1);
//    img2.copyTo(temp2);
//    drawMatches(temp1, this->keypoints[0], temp2, this->keypointsFrame, this->goodMatches, imgMatches);
//    imshow(windowName, imgMatches);
//    waitKey(0);
//}

void KpMatcher::getReprojectionMatrixes(const string &intr, const string &extr, const string &k, Mat &intrinsticMatrix, Mat &extrinsicMatrix, Mat &kMatrix)
{
    readMatrixFromFile(intr, 3, intrinsticMatrix);
    readMatrixFromFile(extr, 4, extrinsicMatrix);
    readMatrixFromFile(k, 4, kMatrix);
}


void KpMatcher::get2dPointsFromKeypoints(vector<KeyPoint> &keypointsFrame, vector<Vec3f> &points2d)
{
    for(int i = 0; i < keypointsFrame.size(); i++)
    {
        Vec3f temp(keypointsFrame[i].pt.x, keypointsFrame[i].pt.y, 1);
        points2d.push_back(temp);
    }
}

void KpMatcher::getSpatialCoordinates(Mat &intr, Mat &extr, vector<Vec3f> &points2d, vector<Vec3f> &points3d)
{    
    Mat R = Mat::zeros(3, 3, extr.type());
    extr.col(0).copyTo(R.col(0));
    extr.col(1).copyTo(R.col(1));
    extr.col(2).copyTo(R.col(2));

    Mat T = Mat::zeros(3, 1, extr.type());
    extr.col(3).copyTo(T.col(0));

    for(int i = 0; i < points2d.size(); i++)
    {
        Mat resultingPoint = ( Mat( points2d[i] ).t() - T.t() ) * R.inv();
        points3d.push_back( Vec3f(resultingPoint) / T.at<float>(2, 0) );
    }
}

void KpMatcher::pcshow(vector<Vec3f> &points3d)
{
    glfwInit();
    int SCREEN_WIDTH = 1280, SCREEN_HEIGHT = 960;
    GLFWwindow * win = glfwCreateWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "PointCloud", NULL, NULL);
    if( !win )
    {
        glfwTerminate();
    }
    else
    {
        int rotate = 0;
        glfwMakeContextCurrent(win);

        while( !glfwWindowShouldClose(win) )
        {
            if(rotate == 360)
            {
                rotate = 0;
            }
            glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();

            gluPerspective(60, (float)1280/960, 0.01f, 20.0f);
            glMatrixMode(GL_MODELVIEW);
            glLoadIdentity();
            gluLookAt(0, 0, -15, 0, 0, 1, 0, -1, 0);
            glTranslatef(0, 0, +0.5f);
            glRotated(0, 1, 0, 0);
            glRotated(rotate, 0, 1, 0);
            glTranslatef(0, 0, -0.5f);

            glPointSize(2);
            glEnable(GL_DEPTH_TEST);
            glBegin(GL_POINT);
            for(int p = 0; p < points3d.size(); p++)
            {
                glColor3ub( 255, 255, 255 );
                glVertex3f( points3d[p][0], points3d[p][1], abs(points3d[p][2]) );
            }
            glEnd();
            glfwSwapBuffers( win );
            rotate++;
        }
    }
}




//
//private section
//
void KpMatcher::readMatrixFromFile(const string &path, const int cols, Mat &out)
{
    ifstream infile(path.c_str());
    string line;
    while (getline(infile, line))
    {
        stringstream ss(line);
        float a, b, c, d;
        Mat row;
        switch(cols)
        {
        case 3:
            if (ss >> a >> b >> c)
            {
                row = (Mat_<float>(1, 3) << a, b, c);
            }
            break;
        case 4:
            if (ss >> a >> b >> c >> d)
            {
                row = (Mat_<float>(1, 4) << a, b, c, d);
            }
            break;
        }
        out.push_back(row);
    }
}

void KpMatcher::setUpRotationMatrix(Mat in, Mat &rotationMatrix, float angle, float u, float v, float w)
{
    float L = (u*u + v * v + w * w);
    float u2 = u * u;
    float v2 = v * v;
    float w2 = w * w;

    rotationMatrix.at<float>(0, 0) = (u2 + (v2 + w2) * cos(angle)) / L;
    rotationMatrix.at<float>(0, 1) = (u * v * (1 - cos(angle)) - w * sqrt(L) * sin(angle)) / L;
    rotationMatrix.at<float>(0, 2) = (u * w * (1 - cos(angle)) + v * sqrt(L) * sin(angle)) / L;

    rotationMatrix.at<float>(1, 0) = (u * v * (1 - cos(angle)) + w * sqrt(L) * sin(angle)) / L;
    rotationMatrix.at<float>(1, 1) = (v2 + (u2 + w2) * cos(angle)) / L;
    rotationMatrix.at<float>(1, 2) = (v * w * (1 - cos(angle)) - u * sqrt(L) * sin(angle)) / L;

    rotationMatrix.at<float>(2, 0) = (u * w * (1 - cos(angle)) - v * sqrt(L) * sin(angle)) / L;
    rotationMatrix.at<float>(2, 1)= (v * w * (1 - cos(angle)) + u * sqrt(L) * sin(angle)) / L;
    rotationMatrix.at<float>(2, 2) = (w2 + (u2 + v2) * cos(angle)) / L;
}

Point2f* KpMatcher::setupSourcePoints(Mat& img)
{
    Point2f* points = new Point2f[NPOINTS];
    points[0] = Point2f(0.0, 0.0);
    points[1] = Point2f(img.cols, 0.0);
    points[2] = Point2f(img.cols, img.rows);
    points[3] = Point2f(0.0, img.rows);
    return points;
}

Point2f* KpMatcher::setupDestinationPoints(Point2f *pointsSrc, TransformPoints value, int factor)
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

Point2f* KpMatcher::cpyArray(Point2f* in, int size)
{
    Point2f* out = new Point2f[NPOINTS];
    for(int i = 0; i<size; i++)
    {
        out[i] = in[i];
    }
    return out;
}

bool KpMatcher::isInParalellgoram(KeyPoint kp, vector<Point2f> &figure)
{
    //couter-clockwise
    for(int i = figure.size(); i > 0; i--)
    {
        float A = -(figure[i-1].y - figure[i].y);
        float B = figure[i-1].x - figure[i].x;
        float C = -(A * figure[i].x + B * figure[i].y);
        float D = A * kp.pt.x + B * kp.pt.y + C;
        if(D > 0)
        {
            return false;
        }
    }
    return true;
}

void KpMatcher::describeAndDetectInitKeypoints(Mat &img)
{
    vector<KeyPoint> keypointsPre;
    detector->detect(img, keypointsPre);
    vector<Point2f> parallellogram;

    //TODO: read this from file
    Point2f topLeft = Point2f(227, 100);
    Point2f topRight = Point2f(515, 122);
    Point2f bottomRight = Point2f(506, 367);
    Point2f bottomLeft = Point2f(208, 343);
    parallellogram.push_back(bottomLeft);
    parallellogram.push_back(topLeft);
    parallellogram.push_back(topRight);
    parallellogram.push_back(bottomRight);
    parallellogram.push_back(bottomLeft);
    for(int i = 0; i < keypointsPre.size(); i++)
    {
        if(!isInParalellgoram(keypointsPre[i], parallellogram))
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

void KpMatcher::computeDescriptorsInProperOrder(vector<KeyPoint> &kp, Mat &d, Mat &img)
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

void KpMatcher::describeAndDetectTransformedKeypoints(Mat &transformedImg, Mat &transformMatrix)
{
    vector<KeyPoint> initialVector = this->keypoints[0];
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
    this->keypoints.push_back(keypointsPre);

//    drawKeypoints(transformedImg, keypointsPre, transformedImg);
//    imshow("keypointsPre", transformedImg);
//    waitKey(0);

    Mat desc;
    this->descriptor->compute( transformedImg, keypointsPre, desc );
    this->descriptors.push_back(desc);
}
void KpMatcher::findCorners( Mat &image, vector<Point2f> &corners )
{
    Size pattern = Size(5,4);
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
            Mat nextImage = readImage(line, IMREAD_GRAYSCALE);
            resize(nextImage, nextImage, Size(640, 480));
            findCorners(nextImage, nextCorners);
            Mat homographyMatrix = findHomography(initialCorners, nextCorners, RANSAC);
            this->transformationMatrixes.push_back(homographyMatrix);
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
