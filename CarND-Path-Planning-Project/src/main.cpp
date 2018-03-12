#include <fstream>
#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "json.hpp"

#include "helper.hpp"
#include "planner.hpp"
#include "spline.h"

using namespace std;

// for convenience
using json = nlohmann::json;



// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_first_of("}");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}


int main() {
  uWS::Hub h;
  Planner planner;
  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0
  double max_s = 6945.554;

  ifstream in_map_(map_file_.c_str(), ifstream::in);

  string line;
  while (getline(in_map_, line)) {
  	istringstream iss(line);
  	double x;
  	double y;
  	float s;
  	float d_x;
  	float d_y;
  	iss >> x;
  	iss >> y;
  	iss >> s;
  	iss >> d_x;
  	iss >> d_y;
  	map_waypoints_x.push_back(x);
  	map_waypoints_y.push_back(y);
  	map_waypoints_s.push_back(s);
  	map_waypoints_dx.push_back(d_x);
  	map_waypoints_dy.push_back(d_y);
  }

  h.onMessage([&planner, &map_waypoints_x,&map_waypoints_y,&map_waypoints_s,&map_waypoints_dx,&map_waypoints_dy](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    //auto sdata = string(data).substr(0, length);
    //cout << sdata << endl;
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {

      auto s = hasData(data);

      if (s != "") {
        auto j = json::parse(s);
        
        string event = j[0].get<string>();
        
        if (event == "telemetry") {
          // j[1] is the data JSON object
          
          // Main car's localization Data
          double car_x = j[1]["x"];			// in map coordinate
          double car_y = j[1]["y"];			// in map coordinate
          double car_s = j[1]["s"];			// Frenet coordinates - distance along the road, in meter
          double car_d = j[1]["d"];			// Frenet coordinates - distance perpendicular to the road, in meter
          double car_yaw = j[1]["yaw"];		// in degree
          double car_speed = j[1]["speed"];	// in mph

          // Previous path data given to the Planner
          auto previous_path_x = j[1]["previous_path_x"];
          auto previous_path_y = j[1]["previous_path_y"];
          // Previous path's end s and d values 
          double end_path_s = j[1]["end_path_s"];
          double end_path_d = j[1]["end_path_d"];

          // Sensor Fusion Data, a list of all other cars on the same side of the road.
          vector<vector<double> > sensor_fusion_raw = j[1]["sensor_fusion"];
          vector<vector<double> > sensor_fusion;
          // filter out cars in opposite direction
          for (vector<double> car:sensor_fusion_raw)
          {
            if (car[SF_D]>0)
            {
              sensor_fusion.push_back(car);
            }
          }
          json msgJson;
          static bool changing_lane=false;
   
          /* */
          vector<double>ptsx;
          vector<double>ptsy;
          double ref_x = car_x;
          double ref_y = car_y;
          double ref_vel = car_speed;
          double ref_yaw = car_yaw;
          double speed_limit = 48.5;
          auto prev_size = previous_path_x.size();

          /* sensor fusion */       
          // look for cars ahead in the same lane  
          auto nearbyCars = scanCars(sensor_fusion, prev_size, 
                                     car_s, car_s+60, car_d-2, car_d+2);
          // if there is car ahead, slow down
          bool slow_down=nearbyCars.size()>0?true:false;
          
          // look for alternative path
          int switch2lane=-1;
          auto current_lane = getLane(car_d);
          if (slow_down==true)
          {                        
            auto adjLanes = findAdjacentLanes(current_lane);
            int lane_d;
            for (auto lane:adjLanes)
            {
              lane_d = getD(lane);
              auto adjLaneCars=scanCars(sensor_fusion, 0, 
                                     car_s-10, car_s+60, lane_d-2, lane_d+2);
              // if found a clear lane, switching to that lane
              if (adjLaneCars.size()==0)
              {
                switch2lane = lane;
                slow_down = false;
                break;
              }
            }
          }
          
          int target_lane = switch2lane>=0? switch2lane : current_lane;

          /* speed control */
          if (slow_down){
            // de-accelerate
            ref_vel= car_speed-1.5;
            cout<<"slowing down to "<<ref_vel<<endl;
          }
          else{
            // accelerate
            ref_vel = car_speed+1.5;
            // cap at speed limit
            if (ref_vel>=speed_limit){
              ref_vel=speed_limit;
            }
          }
                    
          if (prev_size<2)
          {
            double prev_car_x = car_x - cos(car_yaw);
            double prev_car_y = car_y - sin(car_yaw);
            
            ptsx.push_back(prev_car_x);
            ptsx.push_back(car_x);

            ptsy.push_back(prev_car_y);
            ptsy.push_back(car_y);            
          }
          else
          {
            ref_x = previous_path_x[prev_size-1];
            ref_y = previous_path_y[prev_size-1];
            double ref_x_prev = previous_path_x[prev_size-2];
            double ref_y_prev = previous_path_y[prev_size-2];
            ref_yaw = atan2(ref_y-ref_y_prev, ref_x-ref_x_prev);

            ptsx.push_back(ref_x_prev);
            ptsx.push_back(ref_x);

            ptsy.push_back(ref_y_prev);
            ptsy.push_back(ref_y);
          }

          // plan for trajectory in 60 and 90 meters aheads
          vector<double>s_ahead={60,90};
          double target_d = getD(target_lane);
          if (switch2lane!=-1)
          {
            cout<<"target d:"<<target_d<<" target lane:"<<target_lane \
                << " swith2lane:"<<switch2lane <<" current_lane:"<<current_lane<<endl;
          }
          for (auto s_delta:s_ahead)
          {
            auto xy=getXY(car_s+s_delta, target_d, map_waypoints_s, map_waypoints_x, map_waypoints_y);
            ptsx.push_back(xy[0]);
            ptsy.push_back(xy[1]);
          }

          for (int i=0; i<ptsx.size();i++)
          {
            double shift_x = ptsx[i]-ref_x;
            double shift_y = ptsy[i]-ref_y;
            auto xy = rotate(shift_x, shift_y, -ref_yaw);
            ptsx[i] = xy[0];
            ptsy[i] = xy[1];
          }

          // use spline to plan for smooth trajectory
          tk::spline trajPlotter;
          trajPlotter.set_points(ptsx,ptsy);

          vector<double>next_x_vals;
          vector<double>next_y_vals;

          for (int i=0; i< previous_path_x.size(); i++)
          {
            next_x_vals.push_back(previous_path_x[i]);
            next_y_vals.push_back(previous_path_y[i]);
          }

          double target_x = 60;
          double target_y = trajPlotter(target_x);
          double target_dist = distance(0,0,target_x,target_y);
          double x_add_on = 0;

          for (int i=1; i<=50 - previous_path_x.size(); i++)
          {
            double N = target_dist/(0.02*0.99*mph2meterps(ref_vel));
            double x_point = x_add_on+(target_x)/N;
            double y_point = trajPlotter(x_point);

            x_add_on=x_point;
            
            double x_ref = x_point;
            double y_ref = y_point;

            auto xy = rotate(x_ref, y_ref, ref_yaw);

            x_point = xy[0]+ref_x;
            y_point = xy[1]+ref_y;

            next_x_vals.push_back(x_point);
            next_y_vals.push_back(y_point);
          }

          // TODO: define a path made up of (x,y) points that the car will visit sequentially every .02 seconds
          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;



          auto msg = "42[\"control\","+ msgJson.dump()+"]";

          //this_thread::sleep_for(chrono::milliseconds(1000));
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
          
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
