/*
 * houghT.cpp
 *
 *  Created on: Sep 15, 2012
 *      Author: Alberto Casta�o Bardawil
 *      e-mail: acbardawil@gmail.com
 */

#include "houghT.h"
#include <cmath>
#include <fstream>
#include <windows.h>

// Structure used for filling the MAP
typedef struct {
	int votes;
	std::vector<int> indexes_T;
	std::vector<int> indexes_Q;
} dStruct;


typedef struct{
	std::vector<int> indexesT;
	std::vector<int> indexesQ;
}indexVectors;

typedef std::pair<int, indexVectors> pairVoteIndex;
typedef std::vector < pairVoteIndex > vectorVoteIndex;

// Function used to organize the pairs
/*bool sort_vect(const pairVoteIndex& left, const pairVoteIndex& right){
	return left.first > right.first;
}*/


bool sort_vect(const dStruct& left, const dStruct& right){
	return left.votes > right.votes;
}



houghT::houghT() {

	WidthBinSigma = 100;					// A factor of 2 for scale
	WidthBinTheta = 360;					// Bin size of 30 (pi/6) for Orientation
	minVotes = 5;
}



houghT::~houghT() {
	// TODO Auto-generated destructor stub
}

void houghT::setSize(int w, int h, double scalex, double scaley)
{
	WidthBinU = w*scalex;			// 0.25 times the maximum projected training image...
	WidthBinV = h*scaley;			// ... dimension (using the predicted scale) for location
}

// 16 neighbours
void houghT::hough16Bin(std::vector<cv::KeyPoint>& keypointsT, std::vector<cv::KeyPoint>& keypointsQ, std::vector<int>& indexesT, std::vector<int>& indexesQ, std::vector<int>& indexesTout, std::vector<int>& indexesQout){


			LARGE_INTEGER frequency;
		QueryPerformanceFrequency(&frequency); //Getting frequency of the processor
		LARGE_INTEGER t1, t2, t3, t4, t5, t6;
		double time1, time2, time3, time4, time5;
		


	// Auxiliary variables to store the values for the (mapping factors) deltas of each characteristic of the
	// KeyPoint (u,v, scale_factor, orientation).
	double delta_sigma = 0;
	double delta_theta = 0;
	double delta_u = 0;
	double delta_v = 0;
	// 1st and 2nd Nearest bin variables for each delta
		// bin[0][0-1] Sigma		bin[1][0-1] Theta
		// bin[2][0-1] U			bin[3][0-1] V
	int bin[4][2];

	// Auxiliary variable used to convert from degrees to radians
	double Deg2Rad = CV_PI/180;
	// Auxiliary variable used for finding the second nearest bin
	double aux = 0;

	// Create the Map structure and its iterator.
	std::map<std::vector<int>, dStruct> Data;
	std::map<std::vector<int>, dStruct>::iterator it;

	// Auxiliary variable used to store the temporary KEY to the Map
	std::vector<int> aux_KEY;
	// Auxiliary variable structure used to store the temporary information that is going to be stored in the Map.
	dStruct aux_DATA;
	aux_DATA.votes = 0;

	bool simple = false;

//	std::ofstream file;
//	file.open("bins.txt");
	std::cout<<indexesT.size()<<std::endl;

	for(unsigned int i = 0 ; i<indexesT.size() ; i++){
		QueryPerformanceCounter(&t1);
		// ----------------------------------------------------------------------------
		// STEP 1: Calculate the mapping factors for each match
		// ----------------------------------------------------------------------------
			// Scale factor ratio
		delta_sigma = keypointsQ.at(indexesQ.at(i)).size / keypointsT.at(indexesT.at(i)).size;
			// 2D rotation angle
		delta_theta = keypointsQ.at(indexesQ.at(i)).angle - keypointsT.at(indexesT.at(i)).angle;
			// Translation
		if (simple == true){
			delta_u = keypointsQ.at(indexesQ.at(i)).pt.x - keypointsT.at(indexesT.at(i)).pt.x;
			delta_v = keypointsQ.at(indexesQ.at(i)).pt.y - keypointsT.at(indexesT.at(i)).pt.y;
		}else{
			delta_u = keypointsQ.at(indexesQ.at(i)).pt.x - delta_sigma * (keypointsT.at(indexesT.at(i)).pt.x * (cos(delta_theta * Deg2Rad)) - (keypointsT.at(indexesT.at(i)).pt.y * sin(delta_theta * Deg2Rad)) );
			delta_v = keypointsQ.at(indexesQ.at(i)).pt.y - delta_sigma * (keypointsT.at(indexesT.at(i)).pt.x * (sin(delta_theta * Deg2Rad)) + (keypointsT.at(indexesT.at(i)).pt.y * cos(delta_theta * Deg2Rad)) );
		}
		QueryPerformanceCounter(&t2);
		// ----------------------------------------------------------------------------
		// STEP 2: Find to which bin they (all deltas) belong.
		// ----------------------------------------------------------------------------
		// Scale factor ratio ---------------------------------------------------------
		// Find the nearest bin
		bin[0][0] = floor(log(delta_sigma)/log(2.0));

		aux = pow(WidthBinSigma, bin[0][0]) + (pow(WidthBinSigma, bin[0][0])/2.0);
		// Find the second nearest bin			
		bin[0][1] =  ((delta_sigma - aux) >= 0)? bin[0][0] + 1: bin[0][0] - 1 ;

		// 2D Rotation Angle ----------------------------------------------------------
			// Special case 1: when  (0 <= delta_theta <= 15)
		if ( (0 <= delta_theta) && (delta_theta <= WidthBinTheta/2.0) ){
			bin[1][0] = 0;
			bin[1][1] = 11;
			// Special case 1: when  (-15 <= delta_theta < 0)
		}else if ( (-WidthBinTheta/2 <= delta_theta) && (delta_theta < 0) ){
			bin[1][0] = 11;
			bin[1][1] = 0;
			// Normal case:
		}else{
			// Find the nearest bin
			if (delta_theta >= 0){ // If delta_theta is positive
				bin[1][0] = floor(delta_theta/WidthBinTheta);
			}else{ // If angle is negative
				bin[1][0] = floor((360 + delta_theta)/WidthBinTheta);
			}

			aux = WidthBinTheta * (bin[1][0] + 1/2.0);
			// Find the second nearest bin
			if ((delta_theta - aux) >= 0 ){
				bin[1][1] = bin[1][0] + 1;
			}else{
				bin[1][1] = bin[1][0] - 1;
			}
		}

		// Translation ----------------------------------------------------------------
			// U image axis -----------------------------------------------------------
		// Find the nearest bin
		bin[2][0] = floor(delta_u/WidthBinU);
		aux = WidthBinU * (float(bin[2][0]) + 0.5);
		// Find the second nearest bin
		bin[2][1] =  ((delta_u - aux) >= 0)? bin[2][0] + 1: bin[2][0] - 1 ;
			// V image axis -----------------------------------------------------------
		bin[3][0] = floor(delta_v/WidthBinV);
		aux = WidthBinV * (float(bin[3][0]) + 0.5);
		bin[3][1] =  ((delta_v - aux) >= 0)? bin[3][0] + 1: bin[3][0] - 1 ;
		// Find the nearest bin
		bin[3][0] = floor(delta_v/WidthBinV);
		QueryPerformanceCounter(&t3);
		// ----------------------------------------------------------------------------
		// STEP 3: Search if the Key exists, and update the MAP, for all the combinations of:
		// 		- binSigma1		- binTheta1		- binU1		- binV1
		// 		- binSigma2		- binTheta2		- binU2		- binV2
		//
		//	for a total of 16 possible combinations
		// ----------------------------------------------------------------------------
		

//		for(unsigned int k1 = 0 ; k1<2 ; k1++){
	//		for(unsigned int k2 = 0 ; k2<4 ; k2++)
		//		file<<bin[k1][k2]<<' ';
			//file<<std::endl;		
//		}
			
		for(int j = 0 ; j < 16 ; j++ ) {

			// Create Key
			QueryPerformanceCounter(&t1);
			aux_KEY.push_back(bin[0][(j/8) % 2]);
			aux_KEY.push_back(bin[1][(j/4) % 2]);
			aux_KEY.push_back(bin[2][(j/2) % 2]);
			aux_KEY.push_back(bin[3][j % 2]);
			QueryPerformanceCounter(&t2);
			//Add the configuration to the map
			QueryPerformanceCounter(&t3);
			Data.insert(std::map<std::vector<int>, dStruct>::value_type(aux_KEY, aux_DATA));
			it = Data.find(aux_KEY);
			(*it).second.votes++;
			(*it).second.indexes_T.push_back(indexesT.at(i));
			(*it).second.indexes_Q.push_back(indexesQ.at(i));
			QueryPerformanceCounter(&t4);
				// Clear the vector for future use
			aux_KEY.clear();
		}
		QueryPerformanceCounter(&t4);
			std::cout<<1000.0*(t2.QuadPart - t1.QuadPart)/frequency.QuadPart<<std::endl;
			std::cout<<1000.0*(t3.QuadPart - t2.QuadPart)/frequency.QuadPart<<std::endl;
			std::cout<<1000.0*(t4.QuadPart - t3.QuadPart)/frequency.QuadPart<<std::endl;
			std::cout<<std::endl;
	}

	QueryPerformanceCounter(&t2);
	std::vector<dStruct> OutputData;
	// Organize the information on the Map in a new structure. The information is going to be organized
	// with the number of votes as criteria (descending)
	int max_votes = minVotes-1;
	std::map<std::vector<int>, dStruct>::iterator maxit;

	for(it = Data.begin() ; it != Data.end() ; it++){

		// Threshold for quantity of candidates. Candidate must have more than 5 votes
		if ((*it).second.votes > max_votes){
			maxit = it;
			max_votes = it->second.votes;

//			pairVoteIndex pairVI;
//			pairVI.first = it->second.votes;
			//pairVI.second.indexesQ
			//pairVI.second.indexesQ.clear();
			//pairVI.second.indexesT.clear();
//			std::cout << pairVI.second.indexesQ.size();
//			std::vector<int> auxiliar = it->second.indexes_Q;
			//pairVI.second.indexesQ.clear();
			//pairVI.second.indexesQ = it->second.indexes_Q;
		//	pairVI.second.indexesT = it->second.indexes_T;
			//OutputData.push_back(pairVI);
			//OutputData.push_back(it->second);
		}
	}
	if (max_votes > minVotes-1){
		indexesTout = maxit->second.indexes_T;
		indexesQout = maxit->second.indexes_Q;
	}
	//}
	QueryPerformanceCounter(&t3);

			std::cout<<1000.0*(t2.QuadPart - t1.QuadPart)/frequency.QuadPart<<std::endl;
			std::cout<<1000.0*(t3.QuadPart - t2.QuadPart)/frequency.QuadPart<<std::endl;
			std::cout<<std::endl;
}


//no neighbours
void houghT::houghBin(std::vector<cv::KeyPoint>& keypointsT, std::vector<cv::KeyPoint>& keypointsQ, std::vector<int>& indexesT, std::vector<int>& indexesQ, std::vector<int>& indexesTout, std::vector<int>& indexesQout){
	// Auxiliary variables to store the values for the (mapping factors) deltas of each characteristic of the
	// KeyPoint (u,v, scale_factor, orientation).
	double delta_sigma = 0;
	double delta_theta = 0;
	double delta_u = 0;
	double delta_v = 0;
	// 1st and 2nd Nearest bin variables
		// bin[0]: Sigma		bin[1]: Theta
		// bin[2]: U			bin[3]: V
	int bin[4];

	// Auxiliary variable used to convert from degrees to radians
	double Deg2Rad = CV_PI/180;
	// Auxiliary variable used for finding the second nearest bin
	bool simple = true;

	// Create the Map structure and its iterator.
	std::map<std::vector<int>, dStruct> Data;
	std::map<std::vector<int>, dStruct>::iterator it;

	// Auxiliary variable used to store the temporary KEY to the Map
	std::vector<int> aux_KEY;
	aux_KEY.resize(4);
	// Auxiliary variable structure used to store the temporary information that is going to be stored in the Map.
	dStruct aux_DATA;
	aux_DATA.votes = 0;

	for(unsigned int i = 0 ; i<indexesT.size() ; i++){
		// ----------------------------------------------------------------------------
		// STEP 1: Calculate the mapping factors for each i
		// ----------------------------------------------------------------------------
			// Scale factor ratio
		delta_sigma = keypointsQ.at(indexesQ.at(i)).size / keypointsT.at(indexesT.at(i)).size;
			// 2D rotation angle
		delta_theta = keypointsQ.at(indexesQ.at(i)).angle - keypointsT.at(indexesT.at(i)).angle;
			// Translation
		if (simple == true){
			delta_u = keypointsQ.at(indexesQ.at(i)).pt.x - keypointsT.at(indexesT.at(i)).pt.x;
			delta_v = keypointsQ.at(indexesQ.at(i)).pt.y - keypointsT.at(indexesT.at(i)).pt.y;
		}else{
			delta_u = keypointsQ.at(indexesQ.at(i)).pt.x - delta_sigma * (keypointsT.at(indexesT.at(i)).pt.x * (cos(delta_theta * Deg2Rad)) - (keypointsT.at(indexesT.at(i)).pt.y * sin(delta_theta * Deg2Rad)) );
			delta_v = keypointsQ.at(indexesQ.at(i)).pt.y - delta_sigma * (keypointsT.at(indexesT.at(i)).pt.x * (sin(delta_theta * Deg2Rad)) + (keypointsT.at(indexesT.at(i)).pt.y * cos(delta_theta * Deg2Rad)) );
		}

		// ----------------------------------------------------------------------------
		// STEP 2: Find to which bin they (all deltas) belong.
		// ----------------------------------------------------------------------------
		// Scale factor ratio ---------------------------------------------------------
		// Find the nearest bin
		bin[0] = floor(log(delta_sigma)/log(2.0));
		// 2D Rotation Angle ----------------------------------------------------------
		// Find the nearest bin
		bin[1] =  (delta_theta >= 0)? floor(delta_theta/WidthBinTheta): floor((360 + delta_theta)/WidthBinTheta);

		// Translation ----------------------------------------------------------------
			// U image axis -----------------------------------------------------------
		// Find the nearest bin
		bin[2] = floor(delta_u/WidthBinU);

			// V image axis -----------------------------------------------------------
		// Find the nearest bin
		bin[3] = floor(delta_v/WidthBinV);

		// ----------------------------------------------------------------------------
		// STEP 3: Search if the Key exists, and update the MAP, for the combination:
		// 		- binSigma1		- binTheta1		- binU1		- binV1
		// ----------------------------------------------------------------------------
		// Create Key
		aux_KEY[0] = bin[0];
		aux_KEY[1] = bin[1];
		aux_KEY[2] = bin[2];
		aux_KEY[3] = bin[3];

		//Add the configuration to the map
		Data.insert(std::map<std::vector<int>, dStruct>::value_type(aux_KEY, aux_DATA));
		it = Data.find(aux_KEY);
		(*it).second.votes++;
		(*it).second.indexes_T.push_back(indexesT.at(i));
		(*it).second.indexes_Q.push_back(indexesQ.at(i));
	}

	int max_votes1 = minVotes-1;
	int max_votes2 = minVotes-1;
	std::map<std::vector<int>, dStruct>::iterator maxit1;
	std::map<std::vector<int>, dStruct>::iterator maxit2;
	std::map<std::vector<int>, dStruct>::iterator aux;
	// Saving all the Hough Cluster inliers that have more than 2 votes.
 	for(it = Data.begin() ; it != Data.end() ; it++){
		// Threshold for quantity of candidates. Candidate must have more than 3 votes		
		if ((*it).second.votes > max_votes2){
			if ((*it).second.votes > max_votes1){
				max_votes2 = max_votes1; 
				maxit2 = maxit1;
				max_votes1 = (*it).second.votes; 
				maxit1 = it;
			}else{
				maxit2 = it;
				max_votes2 = it->second.votes;
			}
		}
	}
	if (max_votes1 > minVotes-1){
		indexesTout = maxit1->second.indexes_T;
		indexesQout = maxit1->second.indexes_Q;
	}
	if (max_votes2 > minVotes-1){
		indexesTout.insert(indexesTout.end(), maxit2->second.indexes_T.begin(), maxit2->second.indexes_T.end());
		indexesQout.insert(indexesQout.end(), maxit2->second.indexes_Q.begin(), maxit2->second.indexes_Q.end());
	}	
}


void houghT::houghBin6parms(std::vector<cv::KeyPoint>& keypointsT, std::vector<cv::KeyPoint>& keypointsQ, std::vector<int>& indexesT, std::vector<int>& indexesQ, std::vector<int>& indexesTout, std::vector<int>& indexesQout){

	// Auxiliary variables to store the values for the (mapping factors) deltas of each characteristic of the
	// KeyPoint (u,v, scale_factor, orientation).
	double delta_sigma = 0;
	double delta_theta = 0;
	double delta_u = 0;
	double delta_v = 0;
	double delta_xQ = 0;
	double delta_yQ = 0;

	// 1st and 2nd Nearest bin variables
		// bin[0]: Sigma		bin[1]: Theta
		// bin[2]: U			bin[3]: V
	int bin[6];

	// Auxiliary variable used to convert from degrees to radians
	double Deg2Rad = CV_PI/180;
	// Auxiliary variable used for finding the second nearest bin
	bool simple = true;

	// Create the Map structure and its iterator.
	std::map<std::vector<int>, dStruct> Data;
	std::map<std::vector<int>, dStruct>::iterator it;

	// Auxiliary variable used to store the temporary KEY to the Map
	std::vector<int> aux_KEY;
		aux_KEY.resize(6);
	// Auxiliary variable structure used to store the temporary information that is going to be stored in the Map.
	dStruct aux_DATA;
	aux_DATA.votes = 0;

	for(unsigned int i = 0 ; i<indexesT.size() ; i++){
		// ----------------------------------------------------------------------------
		// STEP 1: Calculate the mapping factors for each i
		// ----------------------------------------------------------------------------
			// Scale factor ratio
		delta_sigma = keypointsQ.at(indexesQ.at(i)).size / keypointsT.at(indexesT.at(i)).size;
			// 2D rotation angle
		delta_theta = keypointsQ.at(indexesQ.at(i)).angle - keypointsT.at(indexesT.at(i)).angle;
			// Translation
		if (simple == true){
			delta_u = keypointsQ.at(indexesQ.at(i)).pt.x - keypointsT.at(indexesT.at(i)).pt.x;
			delta_v = keypointsQ.at(indexesQ.at(i)).pt.y - keypointsT.at(indexesT.at(i)).pt.y;
		}else{
			delta_u = keypointsQ.at(indexesQ.at(i)).pt.x - delta_sigma * (keypointsT.at(indexesT.at(i)).pt.x * (cos(delta_theta * Deg2Rad)) - (keypointsT.at(indexesT.at(i)).pt.y * sin(delta_theta * Deg2Rad)) );
			delta_v = keypointsQ.at(indexesQ.at(i)).pt.y - delta_sigma * (keypointsT.at(indexesT.at(i)).pt.x * (sin(delta_theta * Deg2Rad)) + (keypointsT.at(indexesT.at(i)).pt.y * cos(delta_theta * Deg2Rad)) );
		}
			// Position
		delta_xQ = keypointsQ.at(indexesQ.at(i)).pt.x;
		delta_yQ = keypointsQ.at(indexesQ.at(i)).pt.y;

		// ----------------------------------------------------------------------------
		// STEP 2: Find to which bin they (all deltas) belong.
		// ----------------------------------------------------------------------------
		// Scale factor ratio ---------------------------------------------------------
		// Find the nearest bin
		bin[0] = floor(log(delta_sigma)/log(2.0));

		// 2D Rotation Angle ----------------------------------------------------------
		// Find the nearest bin
		double angVal;
		if (delta_theta >= 0){ // If delta_theta is positive
			 angVal = delta_theta/WidthBinTheta;
		}else{ // If angle is negative
			angVal = (360 + delta_theta)/WidthBinTheta;
		}
		bin[1] = floor(angVal);

		// Translation ----------------------------------------------------------------
			// U image axis -----------------------------------------------------------
		// Find the nearest bin
		bin[2] = floor(delta_u/WidthBinU);

			// V image axis -----------------------------------------------------------
		// Find the nearest bin
		bin[3] = floor(delta_v/WidthBinV);

		// Position ----------------------------------------------------------------
			// U image axis -----------------------------------------------------------
		// Find the nearest bin
		bin[4] = floor(delta_xQ/WidthBinU);

			// V image axis -----------------------------------------------------------
		// Find the nearest bin
		bin[5] = floor(delta_yQ/WidthBinV);

		// ----------------------------------------------------------------------------
		// STEP 3: Search if the Key exists, and update the MAP, for the combination:
		// 		- binSigma1		- binTheta1		- binU1		- binV1
		// ----------------------------------------------------------------------------
		// Create Key
		aux_KEY[0] = bin[0];
		aux_KEY[1] = bin[1];
		aux_KEY[2] = bin[2];
		aux_KEY[3] = bin[3];
		aux_KEY[4] = bin[4];
		aux_KEY[5] = bin[5];

//		std::cout << keypointsT.at(indexesT.at(i)).pt.x << "," << keypointsT.at(indexesT.at(i)).pt.y << "," << keypointsT.at(indexesT.at(i)).size/2 << "," << keypointsT.at(indexesT.at(i)).angle << "," << keypointsQ.at(indexesQ.at(i)).pt.x << "," << keypointsQ.at(indexesQ.at(i)).pt.y << "," << keypointsQ.at(indexesQ.at(i)).size/2 << "," << keypointsQ.at(indexesQ.at(i)).angle << "," << bin[0] << "," << log2(delta_sigma) << "," << bin[1] << "," << angVal << "," << bin[2] << "," << delta_u/WidthBinU << "," << bin[3] << "," << delta_v/WidthBinV << "," << bin[4] << "," << delta_xQ/WidthBinU << "," << bin[5] << "," << delta_yQ/WidthBinV <<  std::endl;
//		std::cout <<"T: " << indexesT.at(i) << " Q: " << indexesQ.at(i) << "\tKP T: " << keypointsT.at(indexesT.at(i)).pt.x << "\t" << keypointsT.at(indexesT.at(i)).pt.y << "\t" << keypointsT.at(indexesT.at(i)).size/2 << "\t" << keypointsT.at(indexesT.at(i)).angle << "\t\tKP Q: " << keypointsQ.at(indexesQ.at(i)).pt.x << "\t" << keypointsQ.at(indexesQ.at(i)).pt.y << "\t" << keypointsQ.at(indexesQ.at(i)).size/2 << "\t" << keypointsQ.at(indexesQ.at(i)).angle << "\t\tBins [sigma, theta, U, V, xQ, yQ]\t" << bin[0] << " (" << log2(delta_sigma) << ") " << bin[1] << " (" << angVal << ") " << bin[2] << " (" << delta_u/WidthBinU << ") " << bin[3] << " (" << delta_v/WidthBinV << ") " << bin[4] << " (" << delta_xQ/WidthBinU << ") " << bin[5] << " (" << delta_yQ/WidthBinV << ") " <<  std::endl;


		//Add the configuration to the map
		Data.insert(std::map<std::vector<int>, dStruct>::value_type(aux_KEY, aux_DATA));
		it = Data.find(aux_KEY);
		(*it).second.votes++;
		(*it).second.indexes_T.push_back(indexesT.at(i));
		(*it).second.indexes_Q.push_back(indexesQ.at(i));
	}

	// Saving all the Hough Cluster inliers that have more than 2 votes.
	for(it = Data.begin() ; it != Data.end() ; it++){
		// Threshold for quantity of candidates. Candidate must have more than 3 votes
		if ((*it).second.votes > 2){	// Saving all the inliers indexes

			indexesTout.insert(indexesTout.end(), (*it).second.indexes_T.begin(), (*it).second.indexes_T.end());
			indexesQout.insert(indexesQout.end(), (*it).second.indexes_Q.begin(), (*it).second.indexes_Q.end());
		}
	}
}

