/*********************************************************************/
/*                                                                   */
/* BASED ON:                                                         */
/* Template-Based Monocular 3D Shape Recovery Using Laplacian Meshes */
/*                                                                   */
/*********************************************************************/

#ifndef _RECONSTRUCTOR_H_
#define _RECONSTRUCTOR_H_

#include <iostream>
#include <iomanip>
#include <armadillo>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>

#include "../epfl/Mesh/LaplacianMesh.h"
#include "../epfl/Linear/ObjectiveFunction.h"
#include "../epfl/Linear/IneqConstrFunction.h"
#include "../epfl/Linear/IneqConstrOptimize.h"
#include "../epfl/Camera.h"

#include "SimulatedAnnealing.h"

class Reconstructor
{
private:
    // Path and file names
    std::string modelCamIntrFile;                        // Camera used for building the model
    std::string modelCamExtFile;
    std::string trigFile;
    std::string refImgFile;
    std::string imCornerFile;
    std::string ctrlPointsFile;

    arma::urowvec ctrPointIds;                      // Set of control points
    arma::urowvec ctrPointCompIds;      			// Set of control points

    Camera modelCamCamera, modelWorldCamera, modelFakeCamera;        // Camera coordinates
                                                                     // Fake camera only for focal length optimization using simulated annealing alghoritm
    cv::Mat refImg, inputImg, img;                      // Reference image and input image

    arma::mat matchesAll, matchesInlier;
    arma::uvec inlierMatchIdxs, matchesInitIdxs;

    arma::mat               bary3DRefKeypoints;
    std::vector<bool> 		existed3DRefKeypoints;		// To indicate a feature points lies on the reference mesh.

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

    std::vector<cv::KeyPoint> kpModel, kpInput;

    IneqConstrOptimize ineqConstrOptimize;	// Due to accumulation of ill-conditioned errors.

    int tempI, tempIterTO;
    bool isFocalAdjustment, isFocalRandom;
    arma::vec reprojErrors;

private:
    void unconstrainedReconstruction(put::Algorithm &opt);
    void unconstrainedReconstruction();

    void updateInternalMatrices(const double& focal);
    bool find3DPointOnMesh(const cv::Point2d& refPoint, arma::rowvec& intersectionPoint);
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
    double adjustFocal(std::vector<double> params);
    void setupErrors();

public:
    LaplacianMesh *refMesh, resMesh;

public:
    Reconstructor();
    ~Reconstructor();

    void init(cv::Mat &image);
    void prepareMatches(std::vector<cv::DMatch> &matches, std::vector<cv::KeyPoint> &kp1, std::vector<cv::KeyPoint> &kp2);
    void deform();
    void savePointCloud(std::string fileName);
    void drawMesh(cv::Mat &inputImg, LaplacianMesh &mesh, std::string fileName);

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
