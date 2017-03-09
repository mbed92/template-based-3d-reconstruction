#include "../include/reconstructor.h"

Reconstructor::Reconstructor()
{
    this->refMesh = NULL;
    modelCamIntrFile	= "cam.intr";
    modelCamExtFile		= "cam.ext";
    trigFile			= "mesh";
    refImgFile			= "model.png";
    imCornerFile		= "im_corners.txt";
    ctrlPointsFile      = "ControlPointIDs.txt";
}

Reconstructor::~Reconstructor()
{
    delete this->refMesh;
}

void Reconstructor::init()
{
    modelWorldCamera.LoadFromFile(this->modelCamIntrFile, this->modelCamExtFile);
    modelCamCamera = Camera( modelWorldCamera.GetA() );

    ctrPointIds.load("ControlPointIDs.txt");

    this->refMesh = new LaplacianMesh();
    refMesh->Load(trigFile);
    refMesh->TransformToCameraCoord(modelWorldCamera);		// Convert the mesh into camera coordinate using world camera
    refMesh->SetCtrlPointIDs(ctrPointIds);
    refMesh->ComputeAPMatrices();
    refMesh->computeFacetNormalsNCentroids();
    resMesh = *refMesh;
}

void Reconstructor::deform()
{

}

void Reconstructor::drawRefMesh(cv::Mat &inputImg)
{
    arma::mat projPoints = this->modelCamCamera.ProjectPoints(this->refMesh->GetVertexCoords());

    // Drawing mesh on image
    const arma::umat& edges = this->resMesh.GetEdges();
    while(true)
    {
        Mat img = inputImg.clone();
        for (int i = 0; i < this->resMesh.GetNEdges(); i++)
        {
            int vid1 = edges(0, i);		// Vertex id 1
            int vid2 = edges(1, i);		// Vertex id 2

            Point2d pt1(projPoints(vid1,0), projPoints(vid1,1));
            Point2d pt2(projPoints(vid2,0), projPoints(vid2,1));

            // Draw a line
            line( inputImg, pt1, pt2, CV_RGB(255, 0 ,0) );
        }

        namedWindow("test", WINDOW_NORMAL);
        imshow("test", img);
        waitKey(0);
    }
}
