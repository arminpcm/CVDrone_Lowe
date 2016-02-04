#pragma once
#include <map>
#include <vector>
#include <queue>
#include <string>
#include <memory>
#include <pthread.h>
#include <sstream>

namespace parallel_pipeline
{
	class job_base
	{
	public:
		virtual ~job_base(){};
		virtual std::ostream& printStream(std::ostream& input);
		friend std::ostream& operator<<(std::ostream& input, job_base& job);
	};
	
	class pipeline;
	class pipe
	{
	private:
		pthread_t threadworker;
		pthread_mutex_t threadmutex;
		pthread_cond_t threadcondition;
		
		std::queue<std::shared_ptr<job_base>> jobs;
		volatile bool terminating;

		static void* startPipe(void*);
		void doThread();
	protected:
		virtual void work(std::shared_ptr<job_base>& job, pipeline* parent) = 0; //Called when processing a job
	public:
		const std::string name;
		static const int maxJobs = 4;
		pipeline* parent; //Pointer to the pipeline this pipe is attached to.

		pipe(const std::string name);
		virtual ~pipe(){};
		virtual void initialize(){} //Called when the worker thread starts
		
		void shutdown();
		void addJob(std::shared_ptr<job_base>& job);
		static void addJob(std::vector<pipe*>& pipes, std::shared_ptr<job_base>& job);
	};

	class pipeline
	{
	private:
		
		pthread_mutex_t popmutex;
		std::map<std::string,std::vector<pipe*>> stages;
		std::vector<std::shared_ptr<job_base>> finishedJobs;
		
	public:
		volatile bool debugging;
		pthread_mutex_t debugmutex;
		std::stringstream debugbuffer;

		pipeline();

		void addStages(int number, pipe*[]);
		void pushJob(const std::string stage, job_base* job);
		void pushJob(const std::string stage, std::shared_ptr<job_base>& job);
		void finishedJob(std::shared_ptr<job_base>& job);
		void getFinishedJobs(std::vector<std::shared_ptr<job_base>>& list); //Returns a finished job.
		void startDebugging();
		void stopDebugging();
	};
}