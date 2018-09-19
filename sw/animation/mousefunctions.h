#ifndef MOUSEFUNCTIONS_H
#define MOUSEFUNCTIONS_H

#include "main.h"

#include <thread>
#include <mutex>
#include "terminalinfo.h"
#include "agent_thread.h"

// A bit of a hack for compatibility with old GLUT
// http://iihm.imag.fr/blanch/software/glut-macosx/
#if !defined(GLUT_WHEEL_UP)
#  define GLUT_WHEEL_UP    3
#  define GLUT_WHEEL_DOWN  4
#  define GLUT_WHEEL_LEFT  5
#  define GLUT_WHEEL_RIGHT 6
#endif

float mx = 0;
float my = 0;
float sx = 0;
float sy = 0;
float zms = 0;
float zscale = 0;
float px, py;
bool paused = false;
float xrat = 0;
float yrat = 0;

void keyboard_callback(unsigned char key, int x, int y)
{
  terminalinfo ti;

  switch (key) {
    case 'c':
      ti.info_msg("Recentering Animation.");
      mx = 0;
      my = 0;
      break;
    case 'z':
      ti.info_msg("Resetting zoom.");
      zms = 0;
      break;
    case 'q':
      ti.info_msg("Quitting Swarmulator.");
      mtx.try_lock();
      program_running = false;
      break;
    case 'p':
      if (!paused) {
        ti.info_msg("Paused. Press `r' to resume or `s' to step forward.");
        mtx.try_lock();
        paused = true;
      }
      break;
    case 'r':
      if (paused) {
        ti.info_msg("Resuming.");
        mtx.unlock();
        paused = false;
        break;
      }
    case 's':
      ti.info_msg("Stepping through. Press `s' to keep stepping forwrad to `r' to resume. ");
      mtx.try_lock();
      mtx.unlock();
      this_thread::sleep_for(chrono::microseconds(1000));
      mtx.lock();
      paused = true;
      break;
    case 'a':
      if (!paused) {
        ti.info_msg("Drawing new agent.");
        mtx.lock(); // TODO: Change so that this creates a new agent regardless of agent type
        create_new_agent(nagents, py, px);
        mtx.unlock();
        break;
      }
    case 'm':
      ti.info_msg("Toggle realtime factor between 1 and the specified value.");
      if (param->simulation_realtimefactor() != 1) {
        realtimefactor = param->simulation_realtimefactor();
        param->simulation_realtimefactor() = 1;
      } else {
        param->simulation_realtimefactor() = realtimefactor;
      }
      break;
  }
}

void mouse_motion_callback(int x, int y)
{
  mx += param->mouse_drag_speed() / zscale * ((float)x / ((float)window_width / xrat) - sx);
  my += param->mouse_drag_speed() / zscale * (-(float)y / ((float)window_height / yrat) - sy);
}

void mouse_motion_callback_passive(int x, int y)
{
  px = ((((float)x / ((float)window_width / xrat)) * 8 / (zscale * xrat)) - 4 / (zscale * xrat)) - mx;
  py = (-((((float)y / ((float)window_height / yrat)) * 8 / (zscale * yrat)) - 4 / (zscale * yrat))) - my;
}

void mouse_click_callback(int button, int state, int x, int y)
{
  if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
    sx = (float)x / ((float)window_width / xrat);
    sy = -(float)y / ((float)window_height / yrat);
  }

  if (button == GLUT_WHEEL_UP) {
    zms += param->mouse_zoom_speed();
  } else if (button == GLUT_WHEEL_DOWN) {
    zms += -param->mouse_zoom_speed();
  }

  // Guard on too much / too little zoom
  if (zms > 9) {
    zms = 9;
  } else if (zms < -90) {
    zms = -90;
  }
}

void mouse_draganddrop()
{
  glutMotionFunc(mouse_motion_callback);
  glutPassiveMotionFunc(mouse_motion_callback_passive);
  glutMouseFunc(mouse_click_callback);
  glutKeyboardFunc(keyboard_callback);
  zscale = -10 / (-10 + zms);
  glTranslatef(mx, my, -10 + zms);
}

#endif /* MOUSEFUNCTIONS_H */