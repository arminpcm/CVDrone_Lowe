#include "parallel_pipeline.h"
#include <iostream>
#include <algorithm>
#include <fstream>

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

namespace parallel_pipeline
{
	std::ostream& job_base::printStream(std::ostream& stm)
	{
		return stm;
	}

	std::ostream& operator<<(std::ostream& stream, job_base& job)
	{
		return job.printStream(stream);
	}

	void* pipe::startPipe(void* p)
	{
		((pipe*)p)->doThread();
		return 0;
	}

	pipe::pipe(const std::string n) : name(n)
	{
		terminating = false;
		pthread_mutex_init(&threadmutex, NULL);
		pthread_cond_init(&threadcondition, NULL);
		pthread_create(&threadworker, NULL, startPipe, this);
	}
	
	void pipe::doThread()
	{
		initialize();

		while(true)
		{
			pthread_mutex_lock(&threadmutex);
				if(jobs.size() == 0)
					pthread_cond_wait(&threadcondition, &threadmutex);

				if(terminating)
				{
					pthread_mutex_unlock(&threadmutex);
					break;
				}

				std::shared_ptr<job_base> job = jobs.front();
				jobs.pop();
			pthread_mutex_unlock(&threadmutex);

			if(parent->debugging)
			{
				LARGE_INTEGER start, end, freq;

				QueryPerformanceFrequency(&freq);
				QueryPerformanceCounter(&start);
				
				work(job, parent);
				
				QueryPerformanceCounter(&end);
				

				pthread_mutex_lock(&parent->debugmutex);
				parent->debugbuffer << name << " " << *job << " " << ((end.QuadPart - start.QuadPart)*1000/freq.QuadPart) << "ms \n";
				pthread_mutex_unlock(&parent->debugmutex);
			}
			else
				work(job, parent);
		}
	}

	void pipe::shutdown()
	{
		terminating = true;

		pthread_cond_broadcast(&threadcondition);
		pthread_join(threadworker, NULL);
	}

	void pipe::addJob(std::shared_ptr<job_base>& job)
	{
		pthread_mutex_lock(&threadmutex);
		if(jobs.size() < maxJobs)
		{
			jobs.push(job);
			pthread_cond_broadcast(&threadcondition);
		}
		else
			jobs.back() = job;
		pthread_mutex_unlock(&threadmutex);
	}

	void pipe::addJob(std::vector<pipe*>& pipes, std::shared_ptr<job_base>& job)
	{
		std::vector<pipe*>::iterator lowest = std::min_element(pipes.begin(), pipes.end(), 
			[](pipe* a, pipe* b){return a->jobs.size() < b->jobs.size();}
		);

		(**lowest).addJob(job);
	}

	pipeline::pipeline()
	{
		debugging = false;
		pthread_mutex_init(&popmutex, NULL);
		pthread_mutex_init(&debugmutex, NULL);
	}

	void pipeline::pushJob(const std::string stage, job_base* job)
	{
		std::shared_ptr<job_base> ptr(job, [](job_base* obj){
			delete obj;
		});
		pushJob(stage, ptr);
	}

	void pipeline::pushJob(const std::string stage, std::shared_ptr<job_base>& job)
	{
		std::map<std::string, std::vector<pipe*>>::iterator find = stages.find(stage);
		if(find!=stages.end())
		{
			std::vector<pipe*>& pipes = find->second;
			if(!pipes.empty())
				pipe::addJob(find->second, job);
			else
				std::cerr << "Tried to add job to empty stage: " << stage << std::endl;
		}
		else
			std::cerr << "Tried to add job to non-existent stage: " << stage << std::endl;
	}

	void pipeline::finishedJob(std::shared_ptr<job_base>& job)
	{
		pthread_mutex_lock(&popmutex);
			finishedJobs.push_back(job);
		pthread_mutex_unlock(&popmutex);
	}

	void pipeline::getFinishedJobs(std::vector<std::shared_ptr<job_base>>& list)
	{
		pthread_mutex_lock(&popmutex);

		list.clear();
		std::swap(finishedJobs, list);
		
		pthread_mutex_unlock(&popmutex);
	}
	
	void pipeline::addStages(int number, pipe* data[])
	{
		for(int i = 0; i<number; ++i)
		{
			data[i]->parent = this;
			const std::string& name = data[i]->name;

			std::map<std::string, std::vector<pipe*>>::iterator find = stages.find(name);
			if(find!=stages.end())
			{
				find->second.push_back(data[i]);
			}
			else
			{
				stages[name] = std::vector<pipe*>();
				stages[name].push_back(data[i]);
			}
		}
	}

	void pipeline::startDebugging()
	{
		debugging = true;
	}
	void pipeline::stopDebugging()
	{
		if(debugging)
		{
			std::ofstream file("debug_log.txt");
			file << debugbuffer.rdbuf();
			file.flush();

			debugbuffer.clear();
			debugging = false;
		}
	}
}