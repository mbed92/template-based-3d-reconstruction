/*********************************************************************/
/* BASED ON:                                                         */
/* Template-Based Monocular 3D Shape Recovery Using Laplacian Meshes */
/*********************************************************************/

/*~~~~~~~~~~GENERAL EPFL 3D RECONSTRUCTION ALGHORITM~~~~~~~~~~*/
//  - load 3d mesh from file
//  - do matching between 2D images
//  - pick inliers indexes
//  - reconstruct without constraints (ReconstructPlanarUnconstrIter)
//      - put all correspondences of inliers to the matrix (buildCorrespondenceMatrix)
//      - create MPinit matrix using all vertices and all matches
//      - formulate Laplacian: min(x) ||MP||^2 + wr^2 * ||AP||^2 s.t. ||P|| = 1
//      - get eigen vectors of above matrix
//      - set vertex coordinates (resMesh.SetVertexCoords(paramMat * matC)): eigenvectors * parametrization matrix (reference mesh parametrized by control points) = new vertex coords
//      - update edges (which means to get Z coordinate)
//      - set vertex coordinates again (resMesh.SetVertexCoords(scale * paramMat * matC))
//  - optimize using control points (ReconstructIneqConstr)
//      - TODO

#ifndef _RECONSTRUCTOR_H_
#define _RECONSTRUCTOR_H_

#include <iostream>
#include <time.h>

#include <armadillo>

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d.hpp"

#include "../epfl/Mesh/LaplacianMesh.h"
#include "../epfl/Camera.h"

#include <iomanip>

using namespace std;
using namespace cv;

class Reconstructor
{
private:
    // Path and file names
    string modelCamIntrFile;					// Camera used for building the model
    string modelCamExtFile;
    string trigFile;
    string refImgFile;
    string imCornerFile;
    string ctrlPointsFile;

    arma::urowvec ctrPointIds;      			// Set of control points

    Camera modelCamCamera, modelWorldCamera;    // Camera coordinates
    Mat refImg, inputImg;                       // Reference image and input image

public:
    LaplacianMesh *refMesh, resMesh;    		// Select planer or non-planer reference mesh


public:
    Reconstructor();
    ~Reconstructor();
    void init();
    void deform();
    void drawRefMesh(Mat &inputImg);
};

#endif
