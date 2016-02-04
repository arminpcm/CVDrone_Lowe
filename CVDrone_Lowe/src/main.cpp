#include "ardrone/ardrone.h"
#include "quadrotor_pipeline.h"
#include "Draw.h"

using namespace parallel_pipeline;

// --------------------------------------------------------------------------
// main(Number of arguments, Argument values)
// Description  : This is the entry point of the program.
// Return value : SUCCESS:0  ERROR:-1
// --------------------------------------------------------------------------
int main(int argc, char* argv[])
{
	const std::string Training = "quad2.png";
	quadrotor_matcher::image_template_gray = cv::imread(Training, 0);
	if( !quadrotor_matcher::image_template_gray.data )
	{
		std::cout << "...Couldn't find template image." << std::endl;
		return -1;
	}

	Draw draw;

	quadrotor_affine::initialize_st();

	quadrotor_job::ctrl_gain = 0.0025;
	quadrotor_job::ctrl_sweetspot = 60000.0;
	quadrotor_job::ctrl_gain2 = 1.55;//1.15;
	
	pipe* stages[] = {new quadrotor_image, new quadrotor_sift, new quadrotor_matcher, new quadrotor_affine};

	parallel_pipeline::pipeline quadrotor_pipeline;
	quadrotor_pipeline.addStages(4, stages);
	//quadrotor_pipeline.startDebugging();

	//Send the template into the pipe
	quadrotor_job* newjob = new quadrotor_job;
	newjob->image_gray = quadrotor_matcher::image_template_gray;
	quadrotor_pipeline.pushJob("Sift",newjob);

	// AR.Drone class
	ARDrone ardrone;

	// Initialize
	if (!ardrone.open(/*"192.168.1.3"*/)) {
		printf("Failed to initialize.\n");
		return -1;
	}

	ardrone.setFlatTrim();
	
	// Instructions
	printf("***************************************\n");
	printf("*      CV Drone Tracker Program       *\n");
	printf("***************************************\n");
	printf("*                                     *\n");
	printf("* - Controls -                        *\n");
	printf("*    'Space' -- Takeoff/Landing       *\n");
	printf("*                                     *\n");
	printf("* - Others -                          *\n");
	printf("*    'C'     -- Change camera         *\n");
	printf("*    'Esc'   -- Exit                  *\n");
	printf("*                                     *\n");
	printf("***************************************\n\n");

	{
		cv::Mat image = ardrone.getImage();
		quadrotor_affine::image_center_query.at<double>(0) = (double)(image.cols)/2.0;
		quadrotor_affine::image_center_query.at<double>(1) = (double)(image.rows)/2.0;
		quadrotor_affine::image_center_query.at<double>(2) = 1.0;
	}

	{		
		quadrotor_affine::image_center_training.at<double>(0) = (double)(quadrotor_matcher::image_template_gray.cols)/2.0;
		quadrotor_affine::image_center_training.at<double>(1) = (double)(quadrotor_matcher::image_template_gray.rows)/2.0;
		quadrotor_affine::image_center_training.at<double>(2) = 1.0;
	}

	unsigned long curTime = timeGetTime();
	unsigned long lastTime = curTime;
	unsigned long lastBatteryTime = curTime;
	unsigned long lastControl = curTime;
	int cameraMode = 0;

	cv::Mat savedControl;
	while (1)
	{
		curTime = timeGetTime();

		// Battery
		if(lastBatteryTime < curTime)
		{
			lastBatteryTime = curTime + 10000;
			printf("Battery = %d%%\n", ardrone.getBatteryPercentage());
		}

		// Sleep and get key
		int key = cv::waitKey(30);
		switch(key)
		{
		case 0x1b:
			quadrotor_pipeline.stopDebugging();
			goto quit;
		case ' ':
			if (ardrone.onGround()) 
				ardrone.takeoff();
			else                    
				ardrone.landing();
			break;
		case 'c':
			ardrone.setCamera(++cameraMode%4);
			break;
		}
		
		if(ardrone.willGetNewImage())
		{
			quadrotor_job* newjob = new quadrotor_job;
			newjob->image_color = ardrone.getImage();
			quadrotor_pipeline.pushJob("Image",newjob);
		}
		
		std::vector<std::shared_ptr<job_base>> finished_jobs;
		quadrotor_pipeline.getFinishedJobs(finished_jobs);

		if(!finished_jobs.empty())
		{
			std::shared_ptr<quadrotor_job> job = std::dynamic_pointer_cast<quadrotor_job, job_base>(finished_jobs.back());
			if (job->good_match)
			{
				savedControl = job->control;
				lastControl = curTime+500;
				draw.drawContour(quadrotor_matcher::image_template_gray, job->image_gray, quadrotor_affine::Contour, quadrotor_matcher::keypoints_template, job->keypoints, job->Inliers_T, job->Inliers_Q, job->AT, cv::Scalar(0,255,255), "video", true);
			}
			else
			{
				//cout << "WARNING!  No Affine Transformation could be computed" << endl;
				draw.drawMatches(quadrotor_matcher::image_template_gray, job->image_gray, quadrotor_matcher::keypoints_template, job->keypoints, job->Initial_Matches_T, job->Initial_Matches_Q,"video",true);
			}
		}

		if(curTime < lastControl)
		{
			ardrone.move3D(savedControl.at<double>(0), savedControl.at<double>(1), savedControl.at<double>(2), savedControl.at<double>(3));
		}
	}
	// See you
	quit:;
	ardrone.close();

	return 0;
}