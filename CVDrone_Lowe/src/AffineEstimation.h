/*
 * AffineEstimation.h
 *
 *  Created on: Sep 18, 2012
 *      Author: Alberto Castano Bardawil
 *      e-mail: acbardawil@gmail.com
 */

#include <iostream>
#include <time.h>
//#include <math.h>
//#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#ifndef AFFINEESTIMATION_H_
#define AFFINEESTIMATION_H_

class AffineEstimation {
public:

	/// <summary>
	///	N:  Maximum number of iterations for the algorithm.
	/// </summary>
	int N;

	/// <summary>
	///	eps:  Probability that the selected point is an outlier
	/// </summary>
	double eps;

	/// <summary>
	/// p   Input:  Probability (p) is chosen in a way to ensure that at least one of the random samples of s point is free from outliers.
	/// </summary>
	double p;

	/// <summary>
	/// s:  Sampling set's size
	/// </summary>
	double s;


	/// <summary>
	/// tao:  Residual threshold for the RANSAC algorithm
	/// </summary>
	double tao;

	/// <summary>
	/// Constructor
	/// </summary>
	AffineEstimation();

	/// <summary>
	/// Overloaded Constructor used to initialize RANSAC parameters.
	///	@param eps         Input:  Probability that the selected point is an outlier
	///	@param p           Input:  Probability (p) is chosen in a way to ensure that at least one of the random samples of s point is free from outliers.
	///	@param s           Input:  Sampling set's size
	///	@param input_tao   Input:  Residual threshold for the RANSAC algorithm
	/// </summary>
	AffineEstimation(double eps, double p, double s, double input_tao);

	/// <summary>
	/// Destructor
	/// </summary>
	virtual ~AffineEstimation();

	/// <summary>
	/// Calculates an Affine Estimation with RANSAC
	///	@param imageT    Input:  Training Image
	///	@param imageQ    Input:  Query Image
	///	@param keypointT Input:  Training Image KeyPoints
	///	@param keypointQ Input:  Query Image KeyPoints
	///	@param indexT    Input:  Vector of indexes for the Training Image
	///	@param indexQ    Input:  Vector of indexes for the Query Image
	///	@param indexTout Output: New Vector of indexes for the Training Image
	///	@param indexQput Output: NewVector of indexes for the Query Image
	///	@param affMat    Output: Affine Transformation Matrix. (Aff. Trans. from Training to Query)
	///	@param invaffMat Output: inverse of the Affine Transformation Matrix. (Aff. Trans. from Query to Training)
	/// </summary>
	void calculate(std::vector<cv::KeyPoint>& keypointT, std::vector<cv::KeyPoint>& keypointQ, std::vector<int>& indexT, std::vector<int>& indexQ, std::vector<int>& indexTout, std::vector<int>& indexQout, cv::Mat& affMat, cv::Mat& invaffMat);

	/// <summary>
	/// Calculates an Affine Transformation Matrix between the input points
	///	@param keypointT Input:  Training Image KeyPoints
	///	@param keypointQ Input:  Query Image KeyPoints
	///	@param indexT    Input:  Vector of indexes for the Training Image
	///	@param indexQ    Input:  Vector of indexes for the Query Image
	///	@param affMat    Output: Affine Transformation Matrix.
	/// </summary>
	void calculateAffineTransform(std::vector<cv::KeyPoint>& keypointT, std::vector<cv::KeyPoint>& keypointQ, std::vector<int>& indexT, std::vector<int>& indexQ, cv::Mat affMat);

	/// <summary>
		/// Calculates an Affine Estimation with RANSAC, not taking into account the repeated matches
		///	@param keypointsT        Input:  Training Image KeyPoints
		///	@param keypointsQ        Input:  Query Image KeyPoints
		///	@param indexesT          Input:  Vector of match indexes for the Training Image
		///	@param indexesQ          Input:  Vector of match indexes for the Query Image
		///	@param indexesTout       Output: Vector of inlier indexes for the Training Image
		///	@param indexesQout       Output: Vector of inlier indexes for the Query Image
		///	@param outliersTout      Output: Vector of outlier indexes for the Training Image
		///	@param outliersQout      Output: Vector of outlier indexes for the Query Image
		///	@param affMat            Output: Affine Transformation Matrix. (Aff. Trans. from Training to Query)
		///	@param invaffMat         Output: inverse of the Affine Transformation Matrix. (Aff. Trans. from Query to Training)
		/// @param AffMatFlag        Output: Flag indicating if the Affine Transformation is ill conditioned or not
		/// </summary>
		bool calculate(std::vector<cv::KeyPoint>& keypointsT, std::vector<cv::KeyPoint>& keypointsQ, std::vector<int>& indexesT, std::vector<int>& indexesQ, std::vector<int>& indexesTout, std::vector<int>& indexesQout, cv::Mat& affMat);

		/// <summary>
		/// Calculates the inliers, outliers, residuals and consensus of a set of matches and an Affine Transformation Matrix
		///	@param keypointsT        Input:  Training Image KeyPoints
		///	@param keypointsQ        Input:  Query Image KeyPoints
		///	@param indexesT          Input:  Vector of match indexes for the Training Image
		///	@param indexesQ          Input:  Vector of match indexes for the Query Image
		///	@param indexesTout       Output: Vector of inlier indexes for the Training Image
		///	@param indexesQout       Output: Vector of inlier indexes for the Query Image
		///	@param outliersTout      Output: Vector of outlier indexes for the Training Image
		///	@param outliersQout      Output: Vector of outlier indexes for the Query Image
		///	@param affMat            Input:  Affine Transformation Matrix. (Aff. Trans. from Training to Query)
		///	@param invaffMat         Input:  inverse of the Affine Transformation Matrix. (Aff. Trans. from Query to Training)
		/// </summary>
		int consensus(std::vector<cv::KeyPoint>& keypointsT, std::vector<cv::KeyPoint>& keypointsQ, std::vector<int>& indexesT, std::vector<int>& indexesQ, std::vector<int>& indexesTout, std::vector<int>& indexesQout, std::vector<int>& outliersTout, std::vector<int>& outliersQout, cv::Mat& affMat, cv::Mat& invaffMat);
};

#endif /* AFFINEESTIMATION_H_ */
