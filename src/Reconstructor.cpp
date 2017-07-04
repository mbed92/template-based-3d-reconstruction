#include "../include/Reconstructor.h"

const double Reconstructor::ROBUST_SCALE = 2.5;
const int Reconstructor::DETECT_THRES = 100;

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

void Reconstructor::init(Mat &image)
{
    modelWorldCamera.LoadFromFile(this->modelCamIntrFile, this->modelCamExtFile);

    //generic focal length
    arma::mat A = modelWorldCamera.GetA();
    modelCamCamera = Camera( A );
    modelFakeCamera = Camera( A );

    //setup mesh
    ctrPointIds.load("ControlPointIDs.txt");
    this->refMesh = new LaplacianMesh();
    refMesh->Load(trigFile);
    refMesh->TransformToCameraCoord(modelWorldCamera);		// Convert the mesh into camera coordinate using world camera
    refMesh->SetCtrlPointIDs(ctrPointIds);
    refMesh->ComputeAPMatrices();
    refMesh->computeFacetNormalsNCentroids();
    resMesh = *refMesh;

    //setup internal matrices
    const arma::mat& bigAP = this->refMesh->GetBigAP();
    this->APtAP = bigAP.t() * bigAP;

    //setup internal variables
    nUncstrIters	= 5;
    radiusInit 		= 5          * pow(ROBUST_SCALE, nUncstrIters-1);
    wrInit	  		= 525        * pow(ROBUST_SCALE, nUncstrIters-1);
    image.copyTo(img);

    //specify control variables
    this->isFocalAdjustment = true;
    this->isFocalRandom = false;
}

void Reconstructor::deform()
{
    try
    {
        // for focal length optimization
        if(this->isFocalAdjustment)
        {
            Alghoritm* opt;
            std::string optimizer("SA");

            if(optimizer == "SA")
            {
                opt = createSA(400, 600, 50, 1);
                opt->Init();

                //step 1
                unconstrainedReconstruction(*opt);
            }
        }
        else
        {
            //step 1
            unconstrainedReconstruction();
        }

        //step 2
        // ============== Constrained Optimization =======================
        this->matchesInlier = this->matchesAll.rows(this->inlierMatchIdxs);
        if (this->inlierMatchIdxs.n_rows > DETECT_THRES)
        {
            arma::vec cInit = reshape( resMesh.GetCtrlVertices(), resMesh.GetNCtrlPoints()*3, 1 );	// x1 x2..y1 y2..z1 z2..
            ReconstructIneqConstr(cInit);
        }
    }
    catch(exception &e)
    {
        cerr << e.what() << endl;
    }
}

// in this application vector has only one value -> focal length
double Reconstructor::adjustFocal(vector<double> params)
{
    arma::vec tempErrors;
    bool isOk = params.size() == 1 ? true : false;
    if(isOk)
    {
        this->modelFakeCamera.setFocal(params[0]);
    }

    if (tempI >= tempIterTO)
    {
        tempErrors = this->computeReprojectionErrors(this->resMesh, matchesAll, matchesInitIdxs, true); //last - 1 iteration
    }
    else
    {
        tempErrors = this->computeReprojectionErrors(this->resMesh, matchesAll, inlierMatchIdxs, true); //every other iteration but last two
    }

    return arma::mean(tempErrors);
}

void Reconstructor::setupErrors()
{
    if (tempI >= tempIterTO)
    {
        this->reprojErrors = this->computeReprojectionErrors(this->resMesh, matchesAll, matchesInitIdxs, false); //last - 1 iteration
    }
    else
    {
        this->reprojErrors = this->computeReprojectionErrors(this->resMesh, matchesAll, inlierMatchIdxs, false); //every other iteration but last two
    }
}

void Reconstructor::unconstrainedReconstruction(Alghoritm& opt)
{
    // Input check
    if (this->matchesAll.n_rows == 0) {
        this->inlierMatchIdxs.resize(0);
        return;
    }

    double wr = this->wrInit;
    double radius = this->radiusInit;

    // First, we need to build the correspondent matrix with all given matches to avoid re-computation
    this->buildCorrespondenceMatrix(this->matchesAll);

    // Then compute MPinit. Function reconstructPlanarUnconstr() will use part of MPinit w.r.t currently used matches
    this->MPinit = this->Minit * this->refMesh->GetBigParamMat();

    // vector of i from 0 to <size> - 1, containing <size> elements
    matchesInitIdxs = arma::linspace<arma::uvec>(0, matchesAll.n_rows - 1, matchesAll.n_rows);

    // Currently used matches represented by their indices. Initially, use all matches: [0,1,2..n-1]
    this->inlierMatchIdxs = matchesInitIdxs;

    //iterate
    for (int i = 0; i < nUncstrIters; i++)
    {
        this->reconstructPlanarUnconstr(this->inlierMatchIdxs, wr, this->resMesh);

        // If it is the final iteration, break and don't update "inlierMatchIdxs" or "weights", "radius"
        if (i == nUncstrIters - 1) {
            break;
        }

        // Otherwise, remove outliers
        this->tempI = i;
        this->tempIterTO = nUncstrIters - 2;
        setupErrors();

        // check if NaN
        if(arma::mean(reprojErrors) == arma::mean(reprojErrors))
        {
            // run SA alghoritm
            opt.Run((this));
            opt.Reset(this->modelCamCamera.getFocal());
        }

        // get inliers
        arma::uvec idxs = find( reprojErrors < radius );
        if ( idxs.n_elem == 0 )
            break;

        if (i >= tempIterTO)
        {
            inlierMatchIdxs = matchesInitIdxs.elem( idxs ); // (last - 1) iteration
        }
        else
        {
            inlierMatchIdxs = inlierMatchIdxs.elem( idxs ); // every other iteration but last two
        }

        // Update parameters
        wr		= wr 	 / ROBUST_SCALE;
        radius	= radius / ROBUST_SCALE;

        std::cout << "f(" << opt.GetOptimalParameter()[0] << ") = " << opt.GetOptimalSolution() << std::endl;
    }
}

void Reconstructor::unconstrainedReconstruction()
{
    // Input check
    if (this->matchesAll.n_rows == 0) {
        this->inlierMatchIdxs.resize(0);
        return;
    }

    double wr = this->wrInit;
    double radius = this->radiusInit;
    arma::vec reprojErrors;

    // First, we need to build the correspondent matrix with all given matches to avoid re-computation
    this->buildCorrespondenceMatrix(this->matchesAll);

    // Then compute MPinit. Function reconstructPlanarUnconstr() will use part of MPinit w.r.t currently used matches
    this->MPinit = this->Minit * this->refMesh->GetBigParamMat();

    // vestor of i from 0 to <size> - 1, containing <size> elements
    matchesInitIdxs = arma::linspace<arma::uvec>(0, matchesAll.n_rows - 1, matchesAll.n_rows);

    // Currently used matches represented by their indices. Initially, use all matches: [0,1,2..n-1]
    this->inlierMatchIdxs = matchesInitIdxs;

    //iterate
    for (int i = 0; i < nUncstrIters; i++)
    {
        this->tempI = i;
        this->tempIterTO = nUncstrIters - 2;

        this->reconstructPlanarUnconstr(this->inlierMatchIdxs, wr, this->resMesh);

        // If it is the final iteration, break and don't update "inlierMatchIdxs" or "weights", "radius"
        if (i == nUncstrIters - 1) {
            break;
        }

        // Otherwise, remove outliers
        setupErrors();

        // get inliers
        arma::uvec idxs = find( reprojErrors < radius );
        if ( idxs.n_elem == 0 )
            break;

        if (i >= tempIterTO)
        {
            inlierMatchIdxs = matchesInitIdxs.elem( idxs ); // (last - 1) iteration
        }
        else
        {
            inlierMatchIdxs = inlierMatchIdxs.elem( idxs ); // every other iteration but last two
        }

        // Update parameters
        wr		= wr 	 / ROBUST_SCALE;
        radius	= radius / ROBUST_SCALE;
    }
}

void Reconstructor::updateInternalMatrices(const double &focal)
{
    if(focal > 0)
    {
        this->modelCamCamera.setFocal(focal);
        this->buildCorrespondenceMatrix(this->matchesAll);
        this->MPinit = this->Minit * this->refMesh->GetBigParamMat();
    }
}

void Reconstructor::ReconstructIneqConstr( const arma::vec& cInit )
{
    const arma::mat& paramMat = this->refMesh->GetParamMatrix();
    const arma::mat& bigP		= this->refMesh->GetBigParamMat();

    static bool isFirstFrame = true;
    static arma::vec cOptimal = reshape(this->refMesh->GetVertexCoords().rows(this->refMesh->GetCtrlPointIDs()), this->refMesh->GetNCtrlPoints()*3, 1 );

    // Objective function: use MPwAP which was already computed in unconstrained reconstruction
    ObjectiveFunction *objtFunction;

    if ( this->useTemporal && !isFirstFrame  ) {
        objtFunction = new ObjectiveFunction( this->GetMPwAP(), this->timeSmoothAlpha, cOptimal );
    } else {
        objtFunction = new ObjectiveFunction( this->GetMPwAP() );
    }

    // Constrained function
    IneqConstrFunction cstrFunction( bigP, this->refMesh->GetEdges(), this->refMesh->GetEdgeLengths() );

    if ( this->usePrevFrameToInit && !isFirstFrame )
        cOptimal = ineqConstrOptimize.OptimizeLagrange(cOptimal, *objtFunction, cstrFunction);
    else
        cOptimal = ineqConstrOptimize.OptimizeLagrange(cInit, *objtFunction, cstrFunction);

    arma::mat cOptimalMat = reshape(cOptimal, this->refMesh->GetNCtrlPoints(), 3);
    if ( cOptimalMat(0,2) < 0 )
    {
        // Change the sign if the reconstruction is behind the camera.
        // This happens because we take cOptimal as initial value for constrained optimization.
        cOptimalMat = -cOptimalMat;
    }

    // Update vertex coordinates
    this->resMesh.SetVertexCoords( paramMat * cOptimalMat );

    isFirstFrame = false;
    delete objtFunction;
}


arma::vec Reconstructor::computeReprojectionErrors( const TriangleMesh& trigMesh, const arma::mat& matchesInit, const arma::uvec& currentMatchIdxs, bool useFakeCamera )
{
    const arma::mat& vertexCoords = trigMesh.GetVertexCoords();
    int nMatches = currentMatchIdxs.n_rows;

    arma::vec errors(nMatches);		// Errors of all matches

    for (int i = 0; i < nMatches; i++)
    {
        // Facet (3 vertex IDs) that contains the matching point
        int idx  = currentMatchIdxs(i);
        int vId1 = (int)matchesInit(idx, 0);
        int vId2 = (int)matchesInit(idx, 1);
        int vId3 = (int)matchesInit(idx, 2);

        // 3D vertex coordinates
        const arma::rowvec& vertex1Coords = vertexCoords.row(vId1);
        const arma::rowvec& vertex2Coords = vertexCoords.row(vId2);
        const arma::rowvec& vertex3Coords = vertexCoords.row(vId3);

        double bary1 = matchesInit(idx, 3);
        double bary2 = matchesInit(idx, 4);
        double bary3 = matchesInit(idx, 5);

        // 3D feature point
        arma::rowvec point3D = bary1*vertex1Coords + bary2*vertex2Coords + bary3*vertex3Coords;

        arma::vec matchingPoint(2);
        matchingPoint(0) = matchesInit(idx, 6);
        matchingPoint(1) = matchesInit(idx, 7);

        // TODO: Implement this function instead of call projecting function for a single point. This can save expense of function calls
        // Projection
        arma::vec point2D;
        if(useFakeCamera)
        {
            point2D = this->modelFakeCamera.ProjectAPoint(point3D.t());
        }
        else
        {
            point2D = this->modelCamCamera.ProjectAPoint(point3D.t());
        }

        errors(i) = arma::norm(point2D - matchingPoint, 2);
    }

    return errors;
}

void Reconstructor::prepareMatches(vector<DMatch> &matches, vector<KeyPoint> &kp1, vector<KeyPoint> &kp2)
{
    this->kpModel = kp1;
    this->kpInput = kp2;
    this->matchesAll.resize( matches.size(), 9 );
    int nMatches = 0;

    this->existed3DRefKeypoints.resize(matches.size());
    this->bary3DRefKeypoints.set_size(matches.size(), 6);

    //create ref barry
    for(size_t i = 0; i < matches.size(); i++)
    {
        int idx = matches[i].queryIdx;  //model
        Point2f refPoint( kp1[idx].pt.x, kp1[idx].pt.y );

        arma::rowvec intersectPoint;
        bool isIntersect = this->find3DPointOnMesh(refPoint, intersectPoint);

        existed3DRefKeypoints[i] = isIntersect;
        if(isIntersect)
        {
            this->bary3DRefKeypoints(i, arma::span(0,5)) = intersectPoint;
        }
    }

    //matches - frame keypoints -> potential correspondence
    for(size_t i = 0; i < matches.size(); i++)
    {
        if(existed3DRefKeypoints[i])
        {
            int idx = matches[i].trainIdx;  //input

            //vid1, vid2, vid3, b1, b2, b3
            this->matchesAll(nMatches, arma::span(0,5)) = this->bary3DRefKeypoints.row(i);
            this->matchesAll(nMatches, 6) = kp2[idx].pt.x;  //potential match
            this->matchesAll(nMatches, 7) = kp2[idx].pt.y;

            // ID of the 3D point of the match
            this->matchesAll(nMatches, 8) = i;
            nMatches++;
        }
    }
}

bool Reconstructor::find3DPointOnMesh(const Point2d& refPoint, arma::rowvec& intersectionPoint)
{
    bool found = false;
    arma::vec source = arma::zeros<arma::vec>(3);		// Camera center in camera coordinate

    arma::vec homorefPoint;								// Homogeneous reference image point
    homorefPoint << refPoint.x << refPoint.y << 1;

    arma::vec destination = solve(this->modelCamCamera.GetA(), homorefPoint);

    const arma::umat& facets	  = this->refMesh->GetFacets();             // each row: [vid1 vid2 vid3]
    const arma::mat& vertexCoords = this->refMesh->GetVertexCoords();		// each row: [x y z]

    double minDepth = INFINITY;

    int nFacets = this->refMesh->GetNFacets();
    for (int i = 0; i < nFacets; i++)
    {
        const arma::urowvec &aFacet	= facets.row(i);					// [vid1 vid2 vid3]
        const arma::mat &vABC		= vertexCoords.rows(aFacet);		// 3 rows of [x y z]

        // GET BARRY COORDINATES FROM REFERENCE IMAGE
        // get barry coordinates from [u, v, 1]
        // source - cam center point [0, 0, 0]
        // destination - invK * [u, v, 1]
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        arma::vec bary = findIntersectionRayTriangle(source, destination, vABC);

        if (bary(0) >= 0 && bary(1) >= 0 && bary(2) >= 0)
        {
            double depth = bary(0)*vABC(0,2) + bary(1)*vABC(1,2) + bary(2)*vABC(2,2);
            if (depth < minDepth)
            {
                minDepth = depth;
                intersectionPoint << aFacet(0) << aFacet(1) << aFacet(2) << bary(0) << bary(1) << bary(2) << arma::endr;
                found = true;
            }
        }
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    }

    return found;
}

arma::vec Reconstructor::findIntersectionRayTriangle(const arma::vec& source, const arma::vec& destination, const arma::mat& vABC)
{
    arma::vec direction = destination - source;

    arma::mat A = join_rows(vABC.t(), -direction);					// A = [vABC' -d]
    A = join_cols(A, join_rows(arma::ones(1, 3), arma::zeros(1,1)));	        // A = [A; [1 1 1 0]]

    arma::vec b = join_cols(source, arma::ones(1,1));				// b = [source; 1]

    arma::vec X = solve(A, b);										// X = A \ b

    return X.subvec(0,2);
}

void Reconstructor::buildCorrespondenceMatrix( const arma::mat& matches )
{
    int nMatches	= matches.n_rows;
    int nVertices	= this->refMesh->GetNVertices();

    this->Minit = arma::zeros(2*nMatches, 3*nVertices);
    const arma::mat& A = this->modelCamCamera.GetA();

    for (int i = 0; i < nMatches; i++)
    {
        //M matrix
        const arma::rowvec& vid = matches(i, arma::span(0,2));		// Vertex ids in reference image
        const arma::rowvec& bcs = matches(i, arma::span(3,5));		// Barycentric coordinates in reference image
        const arma::rowvec& uvs = matches(i, arma::span(6,7));		// Image coordinates in input image -> keypoints

        // Mx matrix
        // Vertex coordinates are ordered to be [x1,...,xN, y1,...,yN, z1,...,zN]
        for (int k = 0; k <= 2; k++)
        {
            // First row
            // barrycentric * K2x3 - uv*K3 (3)
            Minit(2*i, vid(0) + k*nVertices) = bcs(0) * ( A(0,k) - uvs(0) * A(2,k) );
            Minit(2*i, vid(1) + k*nVertices) = bcs(1) * ( A(0,k) - uvs(0) * A(2,k) );
            Minit(2*i, vid(2) + k*nVertices) = bcs(2) * ( A(0,k) - uvs(0) * A(2,k) );

            // Second row
            Minit(2*i+1, vid(0) + k*nVertices) = bcs(0) * ( A(1,k) - uvs(1) * A(2,k) );
            Minit(2*i+1, vid(1) + k*nVertices) = bcs(1) * ( A(1,k) - uvs(1) * A(2,k) );
            Minit(2*i+1, vid(2) + k*nVertices) = bcs(2) * ( A(1,k) - uvs(1) * A(2,k) );
        }
    }
}

void Reconstructor::reconstructPlanarUnconstr(const arma::uvec& matchIdxs, double wr, LaplacianMesh& resMesh)
{
    // Parameterization matrix
    const arma::mat& paramMat = this->refMesh->GetParamMatrix();

    // Build the matrix MPwAP = [MP; wr*AP] and compute: (MPwAP)' * (MPwAP)
    this->computeCurrentMatrices( matchIdxs, wr );   //????

    // --------------- Eigen value decomposition --------------------------
    arma::mat V;
    arma::vec s;
    eig_sym(s, V, this->MPwAPtMPwAP);
    const arma::vec& c = V.col(0);
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    arma::mat matC = reshape(c, this->refMesh->GetNCtrlPoints(), 3);

    //    cout << "before" << endl;
    //    drawMesh(img);

    // Update vertex coordinates
    resMesh.SetVertexCoords(paramMat * matC);

    //    cout << "after" << endl;
    //    drawMesh(img);

    // Resulting mesh yields a correct projection on image but it does not preserve lengths.
    // So we need to compute the scale factor and multiply with matC

    // Determine on which side the mesh lies: -Z or Z
    double meanZ = mean(this->resMesh.GetVertexCoords().col(2));
    int globalSign = meanZ > 0 ? 1 : -1;

    const arma::vec& resMeshEdgeLens = this->resMesh.ComputeEdgeLengths();
    const arma::vec& refMeshEdgeLens = this->refMesh->GetEdgeLengths();

    double scale = globalSign * norm(refMeshEdgeLens, 2) / norm(resMeshEdgeLens, 2);

    // Update vertex coordinates
    resMesh.SetVertexCoords( scale * paramMat * matC);

    //    cout << "global sign" << endl;
    //    drawMesh(img);
}

void Reconstructor::computeCurrentMatrices( const arma::uvec& matchIdxs, double wr )
{
    int	nMatches = matchIdxs.n_rows;			// Number of currently used matches

    // Build matrix currentMP by taking some rows of MP corresponding to currently used match indices
    arma::mat currentMP(2 * nMatches, this->MPinit.n_cols);
    for (int i = 0; i < nMatches; i++)
    {
        currentMP.rows(2*i, 2*i+1) = this->MPinit.rows(2*matchIdxs(i), 2*matchIdxs(i) + 1);
    }

    MPwAP 		= join_cols( currentMP, wr*this->refMesh->GetBigAP() );
    MPwAPtMPwAP = currentMP.t() * currentMP + wr*wr * this->APtAP;
}

void Reconstructor::drawMesh(Mat &inputImg, LaplacianMesh &mesh, string fileName)
{
    fileName = fileName + ".png";

    arma::mat projPoints = this->modelCamCamera.ProjectPoints(mesh.GetVertexCoords());

    // Drawing mesh on image
    if (this->inlierMatchIdxs.n_rows > 100)
    {
        const arma::umat& edges = mesh.GetEdges();
        Mat img = inputImg.clone();
        for (int i = 0; i < mesh.GetNEdges(); i++)
        {
            int vid1 = edges(0, i);		// Vertex id 1
            int vid2 = edges(1, i);		// Vertex id 2

            Point2d pt1(projPoints(vid1,0), projPoints(vid1,1));
            Point2d pt2(projPoints(vid2,0), projPoints(vid2,1));

            // Draw a line
            line( img, pt1, pt2, CV_RGB(255, 0 ,0) );
        }

        //namedWindow("test", WINDOW_AUTOSIZE);
        imwrite(fileName, img);
        //imshow("test", img);
        //waitKey(0);
    }
}

void Reconstructor::savePointCloud(string fileName)
{
    fileName = fileName + ".txt";
    ofstream os;
    os.open(fileName.c_str());
    arma::mat toSaveMAt = this->resMesh.GetVertexCoords();
    if(toSaveMAt.n_rows > 0)
    {
        for(size_t i = 0; i < toSaveMAt.n_rows; i++)
        {
            os << std::fixed << std::setprecision(6)
               << toSaveMAt.row(i)[0] << " "
               << toSaveMAt.row(i)[1] << " "
               << toSaveMAt.row(i)[2] << "\r\n";
        }
        os.close();
        cout << "Saved " << fileName << endl << endl;
    }
    else
    {
        cerr << "Cannot save mesh!" << endl << endl;
    }
}
