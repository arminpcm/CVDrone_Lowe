/*
* AffineEstimation.cpp
*
*  Created on: Sep 18, 2012
*      Author: Alberto Castano Bardawil
*      e-mail: acbardawil@gmail.com
*/

#include "AffineEstimation.h"

typedef struct{
	// Consensus of the Affine Transformation Matrix
	int Consensus;
	// Inlier indexes for both Training and Query Images
	std::vector<int> inT;
	std::vector<int> inQ;
	// Outlier indexes for both Training and Query Images
	std::vector<int> outT;
	std::vector<int> outQ;

	cv::Mat AT_TQ;
	//cv::Mat AT_QT;
}indexVectors;

AffineEstimation::AffineEstimation() {
	// TODO Auto-generated constructor stub
	eps = 0.4;
	p = 0.99;
	s = 3;
	N = std::min<int>( 50, ( log(1 - p) / log(1 - pow((1 - eps),s)) ));
	tao = 1.5;
}


AffineEstimation::AffineEstimation(double eps1, double p1, double s1, double input_tao1) {
	eps = eps1;
	p = p1;
	s = s1;
	tao = input_tao1;
	N = ceil( log(1 - p) / log(1 - pow((1 - eps),s)) );
}


AffineEstimation::~AffineEstimation() {
	// TODO Auto-generated destructor stub
}


void AffineEstimation::calculateAffineTransform(std::vector<cv::KeyPoint>& keypointT, std::vector<cv::KeyPoint>& keypointQ, std::vector<int>& indexT, std::vector<int>& indexQ, cv::Mat affMat){

	cv::Mat A = cv::Mat::zeros(2*indexT.size(), 6, CV_64FC1);
	cv::Mat At(6, 2*indexT.size(), CV_64FC1);
	cv::Mat AtA(6, 6, CV_64FC1);
	cv::Mat AtAi(6, 6, CV_64FC1);
	cv::Mat AtAiAt(6, 2*indexT.size(), CV_64FC1);
	cv::Mat AtAiAtb(6, 1, CV_64FC1);
	cv::Mat b(2*indexT.size(), 1, CV_64FC1);

	for(unsigned int i = 0 ; i<indexT.size() ; i++){

		A.row(2*i).col(0).setTo(keypointT.at(indexT.at(i)).pt.x);
		A.row(2*i).col(1).setTo(keypointT.at(indexT.at(i)).pt.y);
		A.row(2*i).col(4).setTo(1);

		A.row(2*i+1).col(2).setTo(keypointT.at(indexT.at(i)).pt.x);
		A.row(2*i+1).col(3).setTo(keypointT.at(indexT.at(i)).pt.y);
		A.row(2*i+1).col(5).setTo(1);

		b.row(2*i).col(0).setTo(keypointQ.at(indexQ.at(i)).pt.x);
		b.row(2*i+1).col(0).setTo(keypointQ.at(indexQ.at(i)).pt.y);
	}

	transpose(A,At);
	AtA = At*A;
	invert(AtA,AtAi);
	AtAiAt = AtAi*At;
	AtAiAtb = AtAiAt*b;

	affMat.row(0).col(0).setTo(AtAiAtb.row(0).col(0));
	affMat.row(0).col(1).setTo(AtAiAtb.row(1).col(0));
	affMat.row(0).col(2).setTo(AtAiAtb.row(4).col(0));
	affMat.row(1).col(0).setTo(AtAiAtb.row(2).col(0));
	affMat.row(1).col(1).setTo(AtAiAtb.row(3).col(0));
	affMat.row(1).col(2).setTo(AtAiAtb.row(5).col(0));
}

//return 1 for successful, 0 for failure
bool AffineEstimation::calculate(std::vector<cv::KeyPoint>& keypointsT, std::vector<cv::KeyPoint>& keypointsQ, std::vector<int>& indexesT, std::vector<int>& indexesQ, std::vector<int>& indexesTout, std::vector<int>& indexesQout, cv::Mat& affMat) {

	// Auxiliary Vectors
	indexVectors RANSAC_candidate;
	indexVectors RANSAC_best;
	RANSAC_candidate.Consensus = 0;
	RANSAC_best.Consensus = 6;

	// Maximum number of inliers threshold
	int T = ceil( (1 - eps)*indexesT.size() );
	if (T < RANSAC_best.Consensus){
		T = RANSAC_best.Consensus;
	}

	// Random indexes used for RANSAC
	int ind1 = 0;
	int ind2 = 0;
	int ind3 = 0;

	// Auxiliary variables used to compute
	cv::Point2f Tpoints[3];
	cv::Point2f Qpoints[3];

	// Auxiliary Affine Transform Matrix, and its inverse
	cv::Mat aff_TQ(2, 3, CV_64FC1);
	cv::Mat aff_QT(2, 3, CV_64FC1);

	// Initialize the random number generator.
	//	srand(time( NULL ));

	// RANSAC Algorithm
	int cont = 0;
	while((cont <= N) && ((int)RANSAC_best.inT.size() < T)){
		/// STEP 1: Randomly select a sample of (s) data points from S and instantiate the model from this subset
		// Choose ind1 randomly, between 0 and (indexesT.size())-1).
		ind1 = (rand() % indexesT.size());

		// Choose ind2 randomly, between 0 and (indexesT.size())-1), and different from ind1.
		ind2 = (rand() % indexesT.size());
		// Check if ind2 is different from ind2.
		while (ind1 == ind2)
		{
			ind2 = (rand() % indexesT.size());
		}

		// Choose ind3 randomly, between 0 and (indexesT.size())-1), and different from ind1 and ind2.
		ind3 = (rand() % indexesT.size());
		// Check if ind2 is different from ind2.
		while ((ind1 == ind3) || (ind2 == ind3))
		{
			ind3 = (rand() % indexesT.size());
		}

		// Enter only if the random generated indexes meet the minimal requirements
		/// Set the 3 points to calculate the  Affine Transform
		Tpoints[0] = keypointsT.at(indexesT.at(ind1)).pt;
		Tpoints[1] = keypointsT.at(indexesT.at(ind2)).pt;
		Tpoints[2] = keypointsT.at(indexesT.at(ind3)).pt;

		Qpoints[0] = keypointsQ.at(indexesQ.at(ind1)).pt;
		Qpoints[1] = keypointsQ.at(indexesQ.at(ind2)).pt;
		Qpoints[2] = keypointsQ.at(indexesQ.at(ind3)).pt;

		/// Get the Affine Transform from Training to Query
		aff_TQ = getAffineTransform( Tpoints, Qpoints );
		if (cv::countNonZero(aff_TQ) > 1)  // AT undefined
		{
			/// Get the Affine Transform from Query to Training -- It is equal to the inverse of the previous
			/// Affine Transform.
			invertAffineTransform(aff_TQ, aff_QT);

			// Clear RANSAC_candidate
			RANSAC_candidate.Consensus = 0;
			RANSAC_candidate.inT.clear();
			RANSAC_candidate.inQ.clear();
			RANSAC_candidate.outT.clear();
			RANSAC_candidate.outQ.clear();
			// Compute the consensus
			RANSAC_candidate.Consensus = consensus(keypointsT, keypointsQ, indexesT, indexesQ, RANSAC_candidate.inT, RANSAC_candidate.inQ, RANSAC_candidate.outT, RANSAC_candidate.outQ, aff_TQ, aff_QT);

			/// STEP 3: If the size of Si (# of inliers) is greater than some threshold (T), re-estimate
			///			the model using all the points of Si and FINISH.
			if (RANSAC_candidate.Consensus > RANSAC_best.Consensus)
			{
				// Fill RANSAC_best
				RANSAC_best.Consensus = RANSAC_candidate.Consensus;
				RANSAC_best.inT = RANSAC_candidate.inT;
				RANSAC_best.inQ = RANSAC_candidate.inQ;
				RANSAC_best.outT = RANSAC_candidate.outT;
				RANSAC_best.outT = RANSAC_candidate.outQ;
				RANSAC_best.AT_TQ = aff_TQ.clone();
			}
		}
		cont++;
	}

	// After all RANSAC iterations \\

	if (RANSAC_best.inT.empty()) // if no inliers
	{ 
		affMat.setTo(0.0);
		return false;
	}
	// if there are inliers ... 
	/// Get the Affine Transform from Training to Query with all points (inliers)
	calculateAffineTransform(keypointsT, keypointsQ, RANSAC_best.inT, RANSAC_best.inQ, affMat);
	if (cv::countNonZero(aff_TQ) > 1) // AT undefined?
		// case it was possible to reestimate AT
	{ 
		/// Get the Affine Transform from Query to Training -- It is equal to the inverse of the previous
		/// Affine Transform.
		cv::Mat invaffMat;
		invertAffineTransform(affMat, invaffMat);
		// Clear RANSAC_candidate
		RANSAC_candidate.inT.clear();
		RANSAC_candidate.inQ.clear();
		RANSAC_candidate.outT.clear();
		RANSAC_candidate.outQ.clear();

		RANSAC_candidate.Consensus = consensus(keypointsT, keypointsQ, indexesT, indexesQ, RANSAC_candidate.inT, RANSAC_candidate.inQ, RANSAC_candidate.outT, RANSAC_candidate.outQ, affMat, invaffMat);

		if (RANSAC_candidate.Consensus >= RANSAC_best.Consensus)
		{
			indexesTout = RANSAC_candidate.inT;
			indexesQout = RANSAC_candidate.inQ;
		} 
		else
		{
			indexesTout = RANSAC_best.inT;
			indexesQout = RANSAC_best.inQ;
			affMat = RANSAC_best.AT_TQ;
		}				
	}
	else
	{
		indexesTout = RANSAC_best.inT;
		indexesQout = RANSAC_best.inQ;
		affMat = RANSAC_best.AT_TQ;
	}				
	return true;
}


int AffineEstimation::consensus(std::vector<cv::KeyPoint>& keypointsT, std::vector<cv::KeyPoint>& keypointsQ, std::vector<int>& indexesT, std::vector<int>& indexesQ, std::vector<int>& indexesTout, std::vector<int>& indexesQout, std::vector<int>& outliersTout, std::vector<int>& outliersQout, cv::Mat& affMat, cv::Mat& invaffMat) {


	cv::Mat T = cv::Mat::ones(3, indexesT.size(), CV_64FC1);
	cv::Mat Q = cv::Mat::ones(3, indexesT.size(), CV_64FC1);

	int Consensus = 0;
	for (unsigned int i=0 ; i<indexesT.size() ; i++){

		T.row(0).col(i).setTo(keypointsT.at(indexesT.at(i)).pt.x);
		T.row(1).col(i).setTo(keypointsT.at(indexesT.at(i)).pt.y);

		Q.row(0).col(i).setTo(keypointsQ.at(indexesQ.at(i)).pt.x);
		Q.row(1).col(i).setTo(keypointsQ.at(indexesQ.at(i)).pt.y);
	}

	cv::Mat r1 = T.rowRange(0,2) - invaffMat*Q;
	cv::Mat r2 = Q.rowRange(0,2) - affMat*T;

	double error;
	for (int i=0 ; i< r1.cols ; i++) {
		error = 0.5*norm(r1.col(i)) + 0.5*norm(r2.col(i));
		if (error < 3*tao){

			Consensus++;
			indexesTout.push_back(indexesT.at(i));
			indexesQout.push_back(indexesQ.at(i));
		}else{
			outliersTout.push_back(indexesT.at(i));
			outliersQout.push_back(indexesQ.at(i));
		}
	}
	return Consensus;
}
