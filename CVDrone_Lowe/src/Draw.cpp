/*
 * Draw.cpp
 *
 *  Created on: Sep 21, 2012
 *      Author: Alberto Casta�o Bardawil
 *      e-mail: acbardawil@gmail.com
 */

#include "Draw.h"

Draw::Draw() {
	// TODO Auto-generated constructor stub
}


Draw::~Draw() {
	// TODO Auto-generated destructor stub
}


void Draw::drawKeyPoints(const cv::Mat& imageT, const cv::Mat& imageQ, std::vector<cv::KeyPoint>& keypointsT, std::vector<cv::KeyPoint>& keypointsQ, std::string WindowName, bool Print){

	// Creating an object to store the merged images Merge(Training, Query)
	int rows;
	int cols = imageT.cols + imageQ.cols;
	if (imageT.rows <= imageQ.rows){
		rows = imageQ.rows;
	}else{
		rows = imageT.rows;
	}
	cv::Mat OutImage(rows, cols, CV_8UC3);			//	Merged Images

	cv::Mat imageTwSIFT(imageT.rows, imageT.rows, OutImage.type());
	cv::Mat imageQwSIFT(imageQ.rows, imageQ.rows, OutImage.type());
	cv::drawKeypoints(	imageT, 										// Original image
						keypointsT, 									// Vector of KeyPoint
						imageTwSIFT, 									// The output image
						cv::Scalar(0,255,255), 							// KeyPoint color
						cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS); 	// Drawing flag
	cv::drawKeypoints(	imageQ,											// Original image
						keypointsQ, 									// Vector of KeyPoints
						imageQwSIFT, 									// The output image
						cv::Scalar(0,255,255), 							// KeyPoint color
						cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);		// Drawing flag


	MergeImages(imageTwSIFT, imageQwSIFT, OutImage);

	if (Print == true){
		print(OutImage, WindowName);
	}
}


void Draw::drawVideoKeyPoints(cv::Mat& image, std::vector<cv::KeyPoint>& keypoints) {

	cv::drawKeypoints(	image, 										// Original image
						keypoints, 										// Vector of KeyPoint
						image,	 									// The output image
						cv::Scalar(0,0,255), 							// KeyPoint color
						cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS); 	// Drawing flag
}


void Draw::MergeImages(const cv::Mat& imageT, const cv::Mat& imageQ, const cv::Mat& outimage) {

	// Auxiliary Mat variables
	cv::Mat roi_LeftImage;		// imageT will be on the left part
	cv::Mat roi_RightImage;		// imageQ will be on the right part, we shift the roi of imageQ.cols on the right

	cv::Mat roi_imageT;
	cv::Mat roi_imageQ;

	// Showing loaded images
	// In one single Window
	roi_LeftImage = outimage(cv::Rect(0,0,imageT.cols,imageT.rows)); 				// imageQ will be on the left part
	roi_RightImage = outimage(cv::Rect(imageT.cols,0,imageQ.cols,imageQ.rows)); 	//ImageT will be on the right part, we shift the roi of imageQ.cols on the right

	roi_imageT = imageT(cv::Rect(0,0,imageT.cols,imageT.rows));
	roi_imageQ = imageQ(cv::Rect(0,0,imageQ.cols,imageQ.rows));

	roi_imageT.copyTo(roi_LeftImage);	// imageT will be on the left of outimage
	roi_imageQ.copyTo(roi_RightImage);	// imageQ will be on the right of outimage
}


void Draw::drawMatches(const cv::Mat& imageT, const cv::Mat& imageQ, std::vector<cv::KeyPoint>& keypointsT, std::vector<cv::KeyPoint>& keypointsQ, std::vector<int>& indexesT, std::vector<int>& indexesQ, std::string WindowName, bool Print){

	// Creating an object to store the merged images Merge(Training, Query)
	int rows;
	int cols = imageT.cols + imageQ.cols;
	if (imageT.rows <= imageQ.rows){
		rows = imageQ.rows;
	}else{
		rows = imageT.rows;
	}
	cv::Mat OutImage(rows, cols, CV_8UC3);			//	Merged Images

	cv::Mat imageTwSIFT(imageT.rows, imageT.rows, OutImage.type());
	cv::Mat imageQwSIFT(imageQ.rows, imageQ.rows, OutImage.type());

	std::vector<cv::KeyPoint> nKP_Q;
	std::vector<cv::KeyPoint> nKP_T;
	for(unsigned int i=0; i<indexesT.size(); ++i){
		nKP_T.push_back(keypointsT.at(indexesT.at(i)));
	}
	for(unsigned int i=0; i<indexesQ.size(); ++i){
		nKP_Q.push_back(keypointsQ.at(indexesQ.at(i)));
	}

	cv::drawKeypoints(	imageT, 										// Original image
						nKP_T, 											// Vector of KeyPoint
						imageTwSIFT, 									// The output image
						cv::Scalar(0,255,255), 							// KeyPoint color
//						cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS); 	// Drawing flag
						cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS); 	// Drawing flag
	cv::drawKeypoints(	imageQ,											// Original image
						nKP_Q, 											// Vector of KeyPoints
						imageQwSIFT, 									// The output image
						cv::Scalar(0,255,255), 							// KeyPoint color
						cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);		// Drawing flag

	MergeImages(imageTwSIFT, imageQwSIFT, OutImage);

	cv::Point2f aux;
	for(unsigned int i=0; i<indexesT.size(); ++i){

		aux.x = keypointsQ.at(indexesQ.at(i)).pt.x + imageT.cols;
		aux.y = keypointsQ.at(indexesQ.at(i)).pt.y;

		line(OutImage, keypointsT.at(indexesT.at(i)).pt, aux, cv::Scalar(0,255,255));
	}

	if (Print == true){
		print(OutImage, WindowName);
	}
}


void Draw::print(const cv::Mat& OutImage, std::string WindowName){

	cv::namedWindow(WindowName);		//Define the window
	cv::imshow(WindowName, OutImage);

	// Saving the image
	//std::string Format = ".jpg";
	//std::string OutputImage = "C:/Users/Tavo/Pictures/Results/" + WindowName + ".jpg";
	//std::cout<<OutputImage<<std::endl;
//	cv::imwrite(OutputImage, OutImage);
}


void Draw::drawConvexHull(const cv::Mat& imageT, const cv::Mat& imageQ, std::vector<cv::KeyPoint>& keypointsT, std::vector<cv::KeyPoint>& keypointsQ, std::vector<int>& indexesT, std::vector<int>& indexesQ, cv::Mat& affMat, cv::Scalar Color, std::string WindowName, bool Print){

	// Creating an object to store the merged images Merge(Training, Query)
	int rows;
	int cols = imageT.cols + imageQ.cols;
	if (imageT.rows <= imageQ.rows){
		rows = imageQ.rows;
	}else{
		rows = imageT.rows;
	}
	cv::Mat OutImage(rows, cols, CV_8UC3);			//	Merged Images

	cv::Mat imageTwSIFT(imageT.rows, imageT.rows, OutImage.type());
	cv::Mat imageQwSIFT(imageQ.rows, imageQ.rows, OutImage.type());

	std::vector<cv::KeyPoint> nKP_Q;
	std::vector<cv::KeyPoint> nKP_T;
	for(unsigned int i=0; i<indexesT.size(); ++i){
		nKP_T.push_back(keypointsT.at(indexesT.at(i)));
	}
	for(unsigned int i=0; i<indexesQ.size(); ++i){
		nKP_Q.push_back(keypointsQ.at(indexesQ.at(i)));
	}

	cv::drawKeypoints(	imageT, 										// Original image
						nKP_T, 											// Vector of KeyPoint
						imageTwSIFT, 									// The output image
						cv::Scalar(0,255,255), 							// KeyPoint color
						cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS); 	// Drawing flag
	cv::drawKeypoints(	imageQ,											// Original image
						nKP_Q, 											// Vector of KeyPoints
						imageQwSIFT, 									// The output image
						cv::Scalar(0,255,255), 							// KeyPoint color
						cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);		// Drawing flag


	MergeImages(imageTwSIFT, imageQwSIFT, OutImage);

	// Auxiliary variables Mat used to compute the mapping from the Training image to the Query image
	cv::Mat uv(3,1, affMat.type());
	cv::Mat uvp(2,1, affMat.type());

	// Auxiliary variables used to compute the convexHull polygons
	std::vector<cv::Point2f> PointsT;
	std::vector<cv::Point2f> PointsQ;

	// Vectors containing the points that define the convexHull polygon
	std::vector<cv::Point2f> hullT;
	std::vector<cv::Point2f> hullQ;

	// Auxiliary variables for plotting
	cv::Point2f point;
	cv::Point2f point1;
	double aux1;
	double aux2;

	for (unsigned int i = 0 ; i < indexesT.size() ; i++){
		PointsT.push_back(keypointsT.at(indexesT.at(i)).pt);
	}

	for (unsigned int i = 0 ; i < indexesT.size() ; i++){

		uv.row(0).col(0).setTo(PointsT.at(i).x);
		uv.row(1).col(0).setTo(PointsT.at(i).y);
		uv.row(2).col(0).setTo(1);

		uvp = affMat*uv;

		aux1 = uvp.at<double>(0,0);
		aux2 = uvp.at<double>(1,0);
		point.x = aux1;
		point.y = aux2;

		PointsQ.push_back(point);
	}

	convexHull(PointsT, hullT);
	convexHull(PointsQ, hullQ);

	// Start plotting
	for(unsigned int i=0; i<hullT.size()-1; ++i){

		line(OutImage, hullT.at(i), hullT.at(i+1), Color);

		point.x = hullQ.at(i).x + imageT.cols;
		point.y = hullQ.at(i).y;
		point1.x = hullQ.at(i+1).x + imageT.cols;
		point1.y = hullQ.at(i+1).y;
		line(OutImage, point, point1, Color);
	}
	line(OutImage, hullT.at(hullT.size()-1), hullT.at(0), Color);
	point.x = hullQ.at(hullQ.size()-1).x + imageT.cols;
	point.y = hullQ.at(hullQ.size()-1).y;
	point1.x = hullQ.at(0).x + imageT.cols;
	point1.y = hullQ.at(0).y;
	line(OutImage, point, point1, Color);

	for(unsigned int i=0; i<indexesT.size(); ++i){
		point.x = PointsQ.at(i).x + imageT.cols;
		point.y = PointsQ.at(i).y;

		line(OutImage, PointsT.at(i), point, Color);
	}

	// Printing
	if (Print == true){
			print(OutImage, WindowName);
	}
}


void Draw::drawContour(const cv::Mat& imageT, const cv::Mat& imageQ, std::vector<cv::Point2f> Contour, std::vector<cv::KeyPoint>& keypointsT, std::vector<cv::KeyPoint>& keypointsQ, std::vector<int>& indexesT, std::vector<int>& indexesQ, cv::Mat& affMat, cv::Scalar Color, std::string WindowName, bool Print){

	// Creating an object to store the merged images Merge(Training, Query)
	int rows;
	int cols = imageT.cols + imageQ.cols;
	if (imageT.rows <= imageQ.rows){
		rows = imageQ.rows;
	}else{
		rows = imageT.rows;
	}
	cv::Mat OutImage(rows, cols, CV_8UC3);			//	Merged Images

	cv::Mat imageTwSIFT(imageT.rows, imageT.rows, OutImage.type());
	cv::Mat imageQwSIFT(imageQ.rows, imageQ.rows, OutImage.type());

	std::vector<cv::KeyPoint> nKP_Q;
	std::vector<cv::KeyPoint> nKP_T;
	for(unsigned int i=0; i<indexesT.size(); ++i){
		nKP_T.push_back(keypointsT.at(indexesT.at(i)));
	}
	for(unsigned int i=0; i<indexesQ.size(); ++i){
		nKP_Q.push_back(keypointsQ.at(indexesQ.at(i)));
	}

	cv::drawKeypoints(	imageT, 										// Original image
						nKP_T, 											// Vector of KeyPoint
						imageTwSIFT, 									// The output image
						cv::Scalar(0,255,255), 							// KeyPoint color
						cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS); 	// Drawing flag
	cv::drawKeypoints(	imageQ,											// Original image
						nKP_Q, 											// Vector of KeyPoints
						imageQwSIFT, 									// The output image
						cv::Scalar(0,255,255), 							// KeyPoint color
						cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);		// Drawing flag


	MergeImages(imageTwSIFT, imageQwSIFT, OutImage);

	// Auxiliary variables Mat used to compute the mapping from the Training image to the Query image
	cv::Mat uv(3,1, affMat.type());
	cv::Mat uvp(2,1, affMat.type());

	// Auxiliary variables used to compute the convexHull polygons
	std::vector<cv::Point2f> PointsQ;

	// Auxiliary variables for plotting
	cv::Point2f point;
	cv::Point2f point1;
	double aux1;
	double aux2;


	for (unsigned int i = 0 ; i < Contour.size() ; i++){

		uv.row(0).col(0).setTo(Contour.at(i).x);
		uv.row(1).col(0).setTo(Contour.at(i).y);
		uv.row(2).col(0).setTo(1);

		uvp = affMat*uv;

		aux1 = uvp.at<double>(0,0);
		aux2 = uvp.at<double>(1,0);
		point.x = aux1;
		point.y = aux2;

		PointsQ.push_back(point);
	}

	// Start plotting
	for(unsigned int i=0; i<(Contour.size()-1); ++i){
		point.x = PointsQ.at(i).x + imageT.cols;
		point.y = PointsQ.at(i).y;

		point1.x = PointsQ.at(i+1).x + imageT.cols;
		point1.y = PointsQ.at(i+1).y;

		line(OutImage, point, point1, Color, 2);
	}
	point.x = PointsQ.at(3).x + imageT.cols;
	point.y = PointsQ.at(3).y;

	point1.x = PointsQ.at(0).x + imageT.cols;
	point1.y = PointsQ.at(0).y;
	line(OutImage, point, point1, Color, 2);

	for(unsigned int i=0; i<Contour.size(); ++i){

		point1.x = PointsQ.at(i).x + imageT.cols;
		point1.y = PointsQ.at(i).y;

		line(OutImage, Contour.at(i), point1, Color);
	}

	// Printing
	if (Print == true){
			print(OutImage, WindowName);
	}
}


void Draw::drawVideoContour(const cv::Mat& imageT, cv::Mat& video, std::vector<cv::Point2f> Contour, std::vector<cv::KeyPoint>& keypointsT, std::vector<cv::KeyPoint>& keypointsQ, std::vector<int>& indexesT, std::vector<int>& indexesQ, cv::Mat& affMat, cv::Scalar Color){

	cv::Mat imageQwSIFT(video.rows, video.rows, CV_8UC3);

	std::vector<cv::KeyPoint> nKP_Q;
	for(unsigned int i=0; i<indexesQ.size(); ++i){
		nKP_Q.push_back(keypointsQ.at(indexesQ.at(i)));
	}

	cv::drawKeypoints(	video,											// Original image
						nKP_Q, 											// Vector of KeyPoints
						video, 									// The output image
						cv::Scalar(0,255,255), 							// KeyPoint color
						cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);		// Drawing flag

	// Auxiliary variables Mat used to compute the mapping from the Training image to the Query image
	cv::Mat uv(3,1, affMat.type());
	cv::Mat uvp(2,1, affMat.type());

	// Auxiliary variables used to compute the convexHull polygons
	std::vector<cv::Point2f> PointsQ;

	// Auxiliary variables for plotting
	cv::Point2f point;
	cv::Point2f point1;
	double aux1;
	double aux2;

	for (unsigned int i = 0 ; i < Contour.size() ; i++){

		uv.row(0).col(0).setTo(Contour.at(i).x);
		uv.row(1).col(0).setTo(Contour.at(i).y);
		uv.row(2).col(0).setTo(1);

		uvp = affMat*uv;

		aux1 = uvp.at<double>(0,0);
		aux2 = uvp.at<double>(1,0);
		point.x = aux1;
		point.y = aux2;

		PointsQ.push_back(point);
	}

	// Start plotting
	for(unsigned int i=0; i<(Contour.size()-1); ++i){
		point.x = PointsQ.at(i).x;
		point.y = PointsQ.at(i).y;

		point1.x = PointsQ.at(i+1).x;
		point1.y = PointsQ.at(i+1).y;

		line(video, point, point1, Color, 3);
	}
	point.x = PointsQ.at(3).x;
	point.y = PointsQ.at(3).y;

	point1.x = PointsQ.at(0).x;
	point1.y = PointsQ.at(0).y;
	line(video, point, point1, Color, 3);
}
