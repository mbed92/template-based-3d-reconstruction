/*********************************************************************/
/*                                                                   */
/* BASED ON:                                                         */
/* Template-Based Monocular 3D Shape Recovery Using Laplacian Meshes */
/*                                                                   */
/*********************************************************************/

/*~~~~~~~~~~ GENERAL EPFL 3D RECONSTRUCTION ALGHORITM (I hope I understand that...) ~~~~~~~~~~*/
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
//      - TODO: describe optimization step

#ifndef _RECONSTRUCTOR_H_
#define _RECONSTRUCTOR_H_

#include <iostream>
#include <iomanip>
#include <armadillo>

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d.hpp"
#include <GLFW/glfw3.h>
#include <GL/glu.h>

#include "../epfl/Mesh/LaplacianMesh.h"
#include "../epfl/Linear/ObjectiveFunction.h"
#include "../epfl/Linear/IneqConstrFunction.h"
#include "../epfl/Linear/IneqConstrOptimize.h"

#include "../epfl/Camera.h"
#include "SimulatedAnnealing.h"

using namespace std;
using namespace cv;

class Alghoritm;

class Reconstructor
{
private:
    // Path and file names
    string modelCamIntrFile;                        // Camera used for building the model
    string modelCamExtFile;
    string trigFile;
    string refImgFile;
    string imCornerFile;
    string ctrlPointsFile;

    arma::urowvec ctrPointIds;                      // Set of control points
    arma::urowvec ctrPointCompIds;      			// Set of control points

    Camera modelCamCamera, modelWorldCamera, modelFakeCamera;        // Camera coordinates
                                                                     // Fake camera only for focal length optimization using simulated annealing alghoritm
    Mat refImg, inputImg, img;                      // Reference image and input image

    arma::mat matchesAll, matchesInlier;
    arma::uvec inlierMatchIdxs, matchesInitIdxs;

    arma::mat  			bary3DRefKeypoints;
    vector<bool> 		existed3DRefKeypoints;		// To indicate a feature points lies on the reference mesh.

    arma::mat Minit;			// Correspondence matrix M given all initial matches: MX = 0
    arma::mat MPinit;			// Precomputed matrix MP in the term ||MPc|| + wr * ||APc||
                                // M, MP will be computed w.r.t input matches

    arma::mat MPwAP;			// Matrix [MP; wr*AP]. Stored for later use in constrained reconstruction

    arma::mat APtAP;			// Precomputed (AP)'*(AP): only need to be computed once
    arma::mat MPwAPtMPwAP;      // Precomputed (MPwAP)' * MPwAP used in eigen value decomposition

    double			wrInit;					// Initial weight of deformation penalty term ||AX||
    double          wcInit;                 // Initial weight of deformation penalty term depending on focal length
    double			radiusInit;				// Initial radius of robust estimator
    int				nUncstrIters;			// Number of iterations for unconstrained reconstruction


    float			timeSmoothAlpha;        // Temporal consistency weight

    bool			useTemporal;			// Use temporal consistency or not
    bool			usePrevFrameToInit;     // Use the reconstruction in the previous frame to
                                            // initalize the constrained recontruction in the current frame

    static const double ROBUST_SCALE;	// Each iteration of unconstrained reconstruction: decrease by this factor
    static const int DETECT_THRES;

    vector<KeyPoint> kpModel, kpInput;

    IneqConstrOptimize ineqConstrOptimize;	// Due to accumulation of ill-conditioned errors.

    int tempI, tempIterTO;
    bool isFocalAdjustment, isFocalRandom;
    arma::vec reprojErrors;

private:
    void unconstrainedReconstruction(Alghoritm* opt);
    void unconstrainedReconstruction();

    void updateInternalMatrices(const double& focal);
    bool find3DPointOnMesh(const Point2d& refPoint, arma::rowvec& intersectionPoint);
    arma::vec findIntersectionRayTriangle(const arma::vec& source, const arma::vec& destination, const arma::mat& vABC);
    void buildCorrespondenceMatrix( const arma::mat& matches );
    void reconstructPlanarUnconstr(const arma::uvec& matchIdxs, double wr , LaplacianMesh &resMesh);
    void computeCurrentMatrices( const arma::uvec& matchIdxs, double wr );
    arma::vec computeReprojectionErrors(const TriangleMesh& trigMesh, const arma::mat& matchesInit, const arma::uvec& currentMatchIdxs , bool useFakeCamera);
    void ReconstructIneqConstr(const arma::vec& cInit);
    const arma::mat& GetMPwAP() const
    {
        return this->MPwAP;
    }
    double adjustFocal(vector<double> params);
    void setupErrors();

public:
    LaplacianMesh *refMesh, resMesh;

public:
    Reconstructor();
    ~Reconstructor();

    void init(Mat &image);
    void prepareMatches(vector<DMatch> &matches, vector<KeyPoint> &kp1, vector<KeyPoint> &kp2);
    void deform();
    void savePointCloud(string fileName);
    void drawMesh(Mat &inputImg, LaplacianMesh &mesh, string fileName);

    void SetNConstrainedIterations( int pNCstrIters) {
        this->ineqConstrOptimize.SetNIterations(pNCstrIters);
    }

    void SetTimeSmoothAlpha( int pTimeSmoothAlpha) {
        this->timeSmoothAlpha = pTimeSmoothAlpha;
    }

    void SetUseTemporal( bool useTemporal ) {
        this->useTemporal = useTemporal;
    }

    void SetUsePrevFrameToInit( bool usePrevFrameToInit ) {
        this->usePrevFrameToInit = usePrevFrameToInit;
    }

    void SetWrInit(double pWrInit) {
        this->wrInit = pWrInit;
    }

    // Set mu value for inequality reconstruction
    void SetMu(double mu) {
        this->ineqConstrOptimize.SetMuValue(mu);
    }

    friend class SimulatedAnnealing;
};

#endif
