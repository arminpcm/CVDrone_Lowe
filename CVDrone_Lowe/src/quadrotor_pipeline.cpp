#include "quadrotor_pipeline.h"

#include "SiftGPU/gl_core_3_0.h"
#include <iostream>

namespace parallel_pipeline
{
	double quadrotor_job::ctrl_gain;
	double quadrotor_job::ctrl_sweetspot;
	double quadrotor_job::ctrl_gain2;

	std::vector<cv::KeyPoint> quadrotor_matcher::keypoints_template;
	cv::Mat quadrotor_matcher::descriptors_template;
	cv::Mat quadrotor_matcher::image_template_gray;

	std::vector<cv::Point2f> quadrotor_affine::Contour;
	cv::Mat quadrotor_affine::image_center_training(3,1, CV_64FC1);
	cv::Mat quadrotor_affine::image_center_query(3,1, CV_64FC1);
	AffineEstimation quadrotor_affine::Aff(0.5, 0.9999, 3.0, 5.0/3.0);//7.5/3.0;
	houghT quadrotor_affine::HT;

	quadrotor_job::quadrotor_job()
	{
		good_match = false;
		control = cv::Mat(4,1, CV_64FC1);
		AT = cv::Mat::zeros(2,3,CV_64FC1);
	}

	std::ostream& quadrotor_job::printStream(std::ostream& stream) const
	{
		return stream;
	}

	void quadrotor_job::calc_quadrotor()
	{
		// Rotation
		cv::Mat proj_center_training = AT*quadrotor_affine::image_center_training;
		double u = quadrotor_affine::image_center_query.at<double>(0) - proj_center_training.at<double>(0);
		control.at<double>(0) = 0.0;	
		control.at<double>(1) = 0.0;
		control.at<double>(2) = 0.0;	 
		control.at<double>(0) = 0.0;

		control.at<double>(3) = ctrl_gain*u;
		double maxspeed1 = 0.5;
		double maxspeed2 = 1;
		control.at<double>(3) = (control.at<double>(3)>maxspeed1)? maxspeed1:control.at<double>(3);
		control.at<double>(3) = (control.at<double>(3)<-maxspeed1)? -maxspeed1:control.at<double>(3);
	/*	std::cout<<"u = "<<u<<std::endl;
		std::cout<<"image_center_query = "<<quadrotor_affine::image_center_query.at<double>(0)<<std::endl;
		std::cout<<"proj_center_training = "<<quadrotor_affine::image_center_training<<std::endl;
		std::cout<<"proj_center_training = "<<proj_center_training.at<double>(0)<<std::endl;;
		std::cout<<"AT = "<<AT<<std::endl;*/
		// Translation
		double area = 0.0;	
		cv::Mat pt1(3,1, CV_64F), pt2(3,1, CV_64F), pt3(3,1, CV_64F);
		//projecting bounding box
		pt1.at<double>(0) = quadrotor_affine::Contour[0].x;
		pt1.at<double>(1) = quadrotor_affine::Contour[0].y;
		pt1.at<double>(2) = 1.0;
		pt2.at<double>(0) = quadrotor_affine::Contour[1].x;
		pt2.at<double>(1) = quadrotor_affine::Contour[1].y;
		pt2.at<double>(2) = 1.0;
		pt3.at<double>(0) = quadrotor_affine::Contour[2].x;
		pt3.at<double>(1) = quadrotor_affine::Contour[2].y;
		pt3.at<double>(2) = 1.0;

		cv::Mat proj_pt1 = AT*pt1;
		cv::Mat proj_pt2 = AT*pt2;
		cv::Mat proj_pt3 = AT*pt3;

		double width = sqrt((proj_pt2.at<double>(0) - proj_pt1.at<double>(0))*(proj_pt2.at<double>(0) - proj_pt1.at<double>(0)) + (proj_pt2.at<double>(1) - proj_pt1.at<double>(1))*(proj_pt2.at<double>(1) - proj_pt1.at<double>(1)));
		double height = sqrt((proj_pt3.at<double>(0) - proj_pt2.at<double>(0))*(proj_pt3.at<double>(0) - proj_pt2.at<double>(0)) + (proj_pt3.at<double>(1) - proj_pt2.at<double>(1))*(proj_pt3.at<double>(1) - proj_pt2.at<double>(1)));
		area = width*height;
		/*	std::cout<<"w"<<width<<std::endl;
		std::cout<<"h"<<height<<std::endl;
		std::cout<<"a"<<area<<std::endl;*/
		//std::cout<<"a"<<area<<std::endl;
		//	std::cout<<"3: " <<proj_center_training<<std::endl;
		control.at<double>(0) = ctrl_gain2*(ctrl_sweetspot - area)/ctrl_sweetspot;
		//std::cout<<"4: " <<control.at<double>(0) <<std::endl;
		
		control.at<double>(0) = (control.at<double>(0)>maxspeed2)? maxspeed2:control.at<double>(0);
		control.at<double>(0) = (control.at<double>(0)<-maxspeed2)? -maxspeed2:control.at<double>(0);
	}

	void quadrotor_job::calc_control()
	{
		std::vector<int> Hough_Matches_T, Hough_Matches_Q;

		for(std::vector<std::vector<cv::DMatch>>::iterator i = matches.begin(); i!=matches.end(); ++i){
			// Check if this descriptor matches with those of the objects
			// Apply NNDR
			if((i->at(0).distance/i->at(1).distance) < 0.9)
			{
				// Vectors of indexes
				Initial_Matches_T.push_back(i->at(0).trainIdx);
				Initial_Matches_Q.push_back(i->at(0).queryIdx);
			}
		}

		quadrotor_affine::HT.houghBin(quadrotor_matcher::keypoints_template, keypoints, Initial_Matches_T, Initial_Matches_Q, Hough_Matches_T, Hough_Matches_Q);

		if ((int)Hough_Matches_T.size()>0 && quadrotor_affine::Aff.calculate(quadrotor_matcher::keypoints_template, keypoints, Hough_Matches_T, Hough_Matches_Q, Inliers_T, Inliers_Q, AT))
		{
			good_match = true;
			calc_quadrotor();
		}
	}

	void quadrotor_image::work(std::shared_ptr<job_base>& ptr, pipeline* pipe)
	{
		std::shared_ptr<quadrotor_job> job = std::dynamic_pointer_cast<quadrotor_job, job_base>(ptr);
		cv::cvtColor(job->image_color, job->image_gray, CV_RGB2GRAY);
		pipe->pushJob("Sift",ptr);
	}

	void quadrotor_sift::initialize()
	{
		//Initialize sift
		char* argv[] = {"-fo", "-1", "-v", "0"/*, "-cuda", "0"*/};
		sift.ParseParam(4, argv);
		sift.CreateContextGL();
		sift.VerifyContextGL();
	}

	void quadrotor_sift::work(std::shared_ptr<job_base>& ptr, pipeline* pipe)
	{
		std::shared_ptr<quadrotor_job> job = std::dynamic_pointer_cast<quadrotor_job, job_base>(ptr);

		cv::Mat& image = job->image_gray;
		sift.RunSIFT(image.cols, image.rows, image.data, GL_LUMINANCE, GL_UNSIGNED_BYTE);

		int sift_count = sift.GetFeatureNum();
		if(sift_count>0)
		{
			cv::Mat descriptors = cv::Mat(sift_count, 128, CV_32F);
			std::vector<cv::KeyPoint> keypoints_cv;
			std::vector<SiftGPU::SiftKeypoint> keypoints(sift_count);

			sift.GetFeatureVector(&keypoints[0], (float*)descriptors.data);

			//Convert keypoints
			{
				keypoints_cv.resize(sift_count);
				std::vector<SiftGPU::SiftKeypoint>::iterator src = keypoints.begin(), 
					end = keypoints.end();
				std::vector<cv::KeyPoint>::iterator dst = keypoints_cv.begin();
				for(; src!=end; ++src, ++dst)
				{
					dst->angle = src->o;
					dst->size = src->s;
					dst->pt.x = src->x;
					dst->pt.y = src->y;
					/*dst->class_id = 0;
					dst->octave = 0;
					dst->response = 0;*/
				}
			}
			
			if(quadrotor_matcher::descriptors_template.empty())
			{
				cv::swap(quadrotor_matcher::descriptors_template, descriptors);
				std::swap(quadrotor_matcher::keypoints_template, keypoints_cv);
			}
			else
			{
				cv::swap(job->descriptors, descriptors);
				std::swap(job->keypoints, keypoints_cv);
				pipe->pushJob("Match", ptr);
			}
		}				
	}

	void quadrotor_matcher::initialize()
	{
		while(descriptors_template.empty());
		descriptors_template_gpu.upload(descriptors_template);
	}

	void quadrotor_matcher::work(std::shared_ptr<job_base>& ptr, pipeline* pipe)
	{
		std::shared_ptr<quadrotor_job> job = std::dynamic_pointer_cast<quadrotor_job, job_base>(ptr);
		cv::gpu::GpuMat descriptors;
		descriptors.upload(job->descriptors);

		matcher.knnMatch(descriptors, descriptors_template_gpu, job->matches, 2);

		pipe->pushJob("Affine", ptr);
	}

	void quadrotor_affine::initialize_st()
	{
		cv::Mat image_center_training(3,1, CV_64FC1);
		cv::Mat image_center_query(3,1, CV_64FC1);
		image_center_training.at<double>(0) = (double)quadrotor_matcher::image_template_gray.cols/2.0;
		image_center_training.at<double>(1) = (double)quadrotor_matcher::image_template_gray.rows/2.0;
		image_center_training.at<double>(2) = 1.0;
		
		quadrotor_affine::HT.setSize(quadrotor_matcher::image_template_gray.cols,
		                         quadrotor_matcher::image_template_gray.rows,
								 1.0, 1.0);

		cv::Point2f point;
		point.x = 0;
		point.y = 0;
		Contour.push_back(point);
		point.x = (float)quadrotor_matcher::image_template_gray.cols;
		point.y = 0;
		Contour.push_back(point);
		point.x = (float)quadrotor_matcher::image_template_gray.cols;
		point.y = (float)quadrotor_matcher::image_template_gray.rows;
		Contour.push_back(point);
		point.x = 0;
		point.y = (float)quadrotor_matcher::image_template_gray.rows;
		Contour.push_back(point);
	}

	void quadrotor_affine::work(std::shared_ptr<job_base>& ptr, pipeline* pipe)
	{
		std::shared_ptr<quadrotor_job> job = std::dynamic_pointer_cast<quadrotor_job, job_base>(ptr);
		job->calc_control();
		pipe->finishedJob(ptr);
	}
}