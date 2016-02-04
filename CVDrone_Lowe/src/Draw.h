/*
 * Draw.h
 *
 *  Created on: Sep 21, 2012
 *      Author: Alberto Casta�o Bardawil
 *      e-mail: acbardawil@gmail.com
 */

#ifndef DRAW_H_
#define DRAW_H_

#include <iostream>
#include <string.h>
#include <math.h>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class Draw {
public:

	Draw();

	virtual ~Draw();

	/// <summary>
	/// Function used to plot the keypoints of two images.
	///	@param imageT      Input:  Training Image
	///	@param imageQ      Input:  Query Image
	///	@param keypointsT  Input:  Training Image KeyPoints
	///	@param keypointsQ  Input:  Query Image KeyPoints
	/// @param WindowName  Input:  Name the user wants to give to the plotting window
	/// @param Print       Input:  Flag used for printing
	/// <summary>
	void drawKeyPoints(const cv::Mat& imageT, const cv::Mat& imageQ, std::vector<cv::KeyPoint>& keypointsT, std::vector<cv::KeyPoint>& keypointsQ, std::string WindowName, bool Print);

	/// <summary>
	/// Function used to plot the keypoints of one image.
	///	@param imageIn     Input:  Input Image
	///	@param imageOut    Output: Output Image
	///	@param keypoints   Input:  Image KeyPoints
	/// <summary>
	void drawVideoKeyPoints(cv::Mat& image, std::vector<cv::KeyPoint>& keypoints);

	/// <summary>
	/// Function used to merge two images into one single image.
	///	@param imageT    Input:  Training  Image
	///	@param imageQ    Input:  Query  Image
	///	@param outimage  Output: Merged Image
	/// </summary>
	void MergeImages(const cv::Mat& imageT, const cv::Mat& imageQ, const cv::Mat& outimage);

	/// <summary>
	/// Function used to draw the matches between two images
	///	@param imageT     Input: Training  Image
	///	@param imageQ     Input: Query  Image
	///	@param keypointsT Input: Training Image KeyPoints
	///	@param keypointsQ Input: Query Image KeyPoints
	///	@param indexesT   Input: Vector of indexes for the Training Image
	///	@param indexesQ   Input: Vector of indexes for the Query Image
	/// @param WindowName Input: Name the user wants to give to the plotting window
	/// @param Print      Input: Flag used for printing or not printing
	/// </summary>
	void drawMatches(const cv::Mat& imageT, const cv::Mat& imageQ, std::vector<cv::KeyPoint>& keypointsT, std::vector<cv::KeyPoint>& keypointsQ, std::vector<int>& indexesT, std::vector<int>& indexesQ, std::string WindowName, bool Print);

	/// <summary>
	/// Function used to print a image with a specific name. And save the image with that name using (*.jpg) format
	///	@param image      Input: Image on which the matches are going to be plotted
	/// @param WindowName Input: Name the user wants to give to the plotting window
	/// </summary>
	void print(const cv::Mat& OutImage, std::string WindowName);

	/// <summary>
	/// Function used to draw the convexHull polygon of the matches and the keypoint-matches
	///	@param imageT     Input: Training  Image
	///	@param imageQ     Input: Query  Image
	///	@param keypointsT Input: Training Image KeyPoints
	///	@param keypointsQ Input: Query Image KeyPoints
	///	@param indexesT   Input: Vector of indexes for the Training Image
	///	@param indexesQ   Input: Vector of indexes for the Query Image
	///	@param affMat     Input: Affine Transformation Matrix. (Aff. Trans. from Training to Query)
	///	@param Color      Input: Desired color for the polygon and the matches. cv::Scalar(255,255,255)
	/// @param WindowName Input: Name the user wants to give to the plotting window
	/// @param Print      Input: Flag used for printing or not printing
	/// </summary>
	void drawConvexHull(const cv::Mat& imageT, const cv::Mat& imageQ, std::vector<cv::KeyPoint>& keypointsT, std::vector<cv::KeyPoint>& keypointsQ, std::vector<int>& indexesT, std::vector<int>& indexesQ, cv::Mat& affMat, cv::Scalar Color, std::string WindowName, bool Print);

	/// <summary>
	/// Function used to draw the contour of an object
	///	@param imageT     Input: Training  Image
	///	@param imageQ     Input: Query  Image
	///	@param Contour    Input: Contour of the object to be mapped from the Training image to the Query image
	///	@param keypointsT Input: Training Image KeyPoints
	///	@param keypointsQ Input: Query Image KeyPoints
	///	@param indexesT   Input: Vector of indexes for the Training Image
	///	@param indexesQ   Input: Vector of indexes for the Query Image
	///	@param affMat     Input: Affine Transformation Matrix. (Aff. Trans. from Training to Query)
	///	@param Color      Input: Desired color for the polygon and the matches. cv::Scalar(255,255,255)
	/// @param WindowName Input: Name the user wants to give to the plotting window
	/// @param Print      Input: Flag used for printing or not printing
	/// </summary>
	void drawContour(const cv::Mat& imageT, const cv::Mat& imageQ, std::vector<cv::Point2f> Contour, std::vector<cv::KeyPoint>& keypointsT, std::vector<cv::KeyPoint>& keypointsQ, std::vector<int>& indexesT, std::vector<int>& indexesQ, cv::Mat& affMat, cv::Scalar Color, std::string WindowName, bool Print);

	/// <summary>
	/// Function used to draw the contour of an object into the video frame
	///	@param imageT     Input: Training  Image
	///	@param imageQ     Input: Query  Image
	///	@param keypointsT Input: Training Image KeyPoints
	///	@param keypointsQ Input: Query Image KeyPoints
	///	@param indexesT   Input: Vector of indexes for the Training Image
	///	@param indexesQ   Input: Vector of indexes for the Query Image
	///	@param affMat     Input: Affine Transformation Matrix. (Aff. Trans. from Training to Query)
	///	@param Color      Input: Desired color for the polygon and the matches. cv::Scalar(255,255,255)
	/// @param WindowName Input: Name the user wants to give to the plotting window
	/// @param Print      Input: Flag used for printing or not printing
	/// </summary>
	void drawVideoContour(const cv::Mat& imageT, cv::Mat& video, std::vector<cv::Point2f> Contour, std::vector<cv::KeyPoint>& keypointsT, std::vector<cv::KeyPoint>& keypointsQ, std::vector<int>& indexesT, std::vector<int>& indexesQ, cv::Mat& affMat, cv::Scalar Color);
};

#endif /* DRAW_H_ */
