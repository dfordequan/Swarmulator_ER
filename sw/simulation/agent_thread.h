#ifndef AGENTTHREAD_H
#define AGENTTHREAD_H

#include <numeric>
#include <string>
#include <functional>
#include <cctype>
#include <algorithm>
#include <condition_variable>

#include "settings.h"

// Include all agents here
#include "particle.h"
#include "wheeled.h"

void run_agent_simulation_step(const int &id)
{
  mtx.lock();
  s.at(id)->update_position();
  mtx.unlock();
  int t_wait = (int)1000000.0 * (1.0 / (param->simulation_updatefreq() * param->simulation_realtimefactor()));
  this_thread::sleep_for(chrono::microseconds(t_wait));
}

void start_agent_simulation(int id)
{
  std::cout << "Agent" << id << "started" << endl;
  while (true)
  {
    run_agent_simulation_step(id);
  }
};

void create_new_agent(int ID, float x0, float y0){
  vector<float> states = {x0, y0, 0.0, 0.0, 0.0, 0.0}; // Initial positions/states
  s.push_back(new AGENT(ID, states, 1.0 / param->simulation_updatefreq()));
  nagents++;
  this_thread::sleep_for(chrono::microseconds(1000));
  thread agent(start_agent_simulation, ID);
  agent.detach();
}
#endif /*AGENTTHREAD_H*/