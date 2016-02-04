#pragma once
#include "parallel_pipeline.h"
#undef max
#undef min
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include "AffineEstimation.h"
#include "houghT.h"
#include "SiftGPU/SiftGPU.h"

namespace parallel_pipeline
{
	class quadrotor_job : public job_base
	{
	public:
		quadrotor_job();

		std::ostream& printStream(std::ostream& input) const;

		cv::Mat image_color, image_gray;
		bool good_match;

		cv::Mat control;
		cv::Mat AT;
		std::vector<int> Initial_Matches_T, Initial_Matches_Q, Inliers_T, Inliers_Q;

		cv::Mat descriptors;
		std::vector<cv::KeyPoint> keypoints;
		std::vector<std::vector<cv::DMatch>> matches;

		void calc_control();
		void calc_quadrotor();

		static double ctrl_gain;
		static double ctrl_sweetspot;
		static double ctrl_gain2;
	};

	class quadrotor_image : public pipe
	{
	protected:
		void work(std::shared_ptr<job_base>& job, pipeline* pipe);
	public:
		quadrotor_image() : pipe("Image"){}
	};

	class quadrotor_sift : public pipe
	{
	private:
		SiftGPU sift;

	protected:
		void work(std::shared_ptr<job_base>& job, pipeline* pipe);

	public:
		quadrotor_sift() : pipe("Sift"){}
		void initialize();
	};

	class quadrotor_matcher : public pipe
	{
	private:
		cv::gpu::BruteForceMatcher_GPU<cv::L2<float>> matcher;
		cv::gpu::GpuMat descriptors_template_gpu;

	protected:
		void work(std::shared_ptr<job_base>& job, pipeline* pipe);

	public:
		quadrotor_matcher() : pipe("Match"){}
		void initialize();

		static std::vector<cv::KeyPoint> keypoints_template;
		static cv::Mat descriptors_template;
		static cv::Mat image_template_gray;
	};

	class quadrotor_affine : public pipe
	{
	private:
	protected:
		void work(std::shared_ptr<job_base>& ptr, pipeline* pipe);
	public:
		quadrotor_affine() : pipe("Affine"){}
		static void initialize_st();
		
		static cv::Mat image_center_training;
		static cv::Mat image_center_query;
		static AffineEstimation Aff;
		static houghT HT;
		static std::vector<cv::Point2f> Contour;
	};
}