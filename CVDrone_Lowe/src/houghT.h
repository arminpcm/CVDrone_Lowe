/*
 * houghT.h
 *
 *  Created on: Sep 15, 2012
 *      Author: Alberto Casta�o Bardawil
 *      e-mail: acbardawil@gmail.com
 */

#ifndef HOUGHT_H_
#define HOUGHT_H_

#include <iostream>
#include <map>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


class houghT {
	public:
		/// Bin width: scale factor
		double WidthBinSigma;
		/// Bin width: orientation
		double WidthBinTheta;
		/// Bin width: u image position
		double WidthBinU;
		/// Bin width: v image position
		double WidthBinV;
		/// minVotes: Minimum number of votes to choose the cluster as a candidate
		int minVotes;

		/// <summary>
		/// Constructor
		///	@param Input: imageT Training Image
		/// </summary>
		houghT();

		/// <summary>
		/// Destructor
		/// </summary>
		virtual ~houghT();

		/// <summary>
		/// Calculates a Hough Transform out of a set of keypoint matches between a training and a query image. The function uses the 2 nearest bins (For a total of 16 combinations)
		///	@param keypointsT  Input:  Training Image KeyPoints
		///	@param keypointsQ  Input:  Query Image KeyPoints
		///	@param indexesT    Input:  Vector of indexes for the Training Image
		///	@param indexesQ    Input:  Vector of indexes for the Query Image
		///	@param indexesTout Output: New Vector of indexes for the Training Image
		///	@param indexesQput Output: New Vector of indexes for the Query Image
		/// </summary>
		void hough16Bin(std::vector<cv::KeyPoint>& keypointsT, std::vector<cv::KeyPoint>& keypointsQ, std::vector<int>& indexesT, std::vector<int>& indexesQ, std::vector<int>& indexesTout, std::vector<int>& indexesQout);

		/// <summary>
		/// Calculates a Hough Transform out of a set of keypoint matches between a training and a query image. The function uses, only the nearest bin (For a total of 1 combination)
		///	@param keypointsT     Input:  Training Image KeyPoints
		///	@param keypointsQ     Input:  Query Image KeyPoints
		///	@param indexesT       Input:  Vector of indexes for the Training Image
		///	@param indexesQ       Input:  Vector of indexes for the Query Image
		///	@param indexesTout    Output: New Vector of indexes for the Training Image
		///	@param indexesQput    Output: New Vector of indexes for the Query Image
		/// </summary>
		void houghBin(std::vector<cv::KeyPoint>& keypointsT, std::vector<cv::KeyPoint>& keypointsQ, std::vector<int>& indexesT, std::vector<int>& indexesQ, std::vector<int>& indexesTout, std::vector<int>& indexesQout);
		void houghBin6parms(std::vector<cv::KeyPoint>& keypointsT, std::vector<cv::KeyPoint>& keypointsQ, std::vector<int>& indexesT, std::vector<int>& indexesQ, std::vector<int>& indexesTout, std::vector<int>& indexesQout);
		void setSize(int w, int h, double scalex, double scaley);
};

#endif /* HOUGHT_H_ */
