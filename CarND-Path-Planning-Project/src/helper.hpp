#ifndef HELPER_HPP
#define HELPER_HPP

enum {SF_ID, SF_X, SF_Y, SF_VX, SF_VY, SF_S, SF_D};

using namespace std;
extern double mph2meterps(double v_mph);
extern constexpr double pi();
extern double deg2rad(double x);
extern double rad2deg(double x);
extern double distance(double x1, double y1, double x2, double y2);
extern int ClosestWaypoint(double x, double y, vector<double> maps_x, vector<double> maps_y);
extern int NextWaypoint(double x, double y, double theta, vector<double> maps_x, vector<double> maps_y);
extern vector<double> getFrenet(double x, double y, double theta, vector<double> maps_x, vector<double> maps_y);
extern vector<double> getXY(double s, double d, vector<double> maps_s, vector<double> maps_x, vector<double> maps_y);
extern double polyeval(vector<double> coeffs, double x);
extern void print_xy(string label, vector<double>x, vector<double>y);
extern vector<double> rotate(double x, double y, double angle);
extern double getD(int lane);
extern int getLane(double d);
extern vector<int> findAdjacentLanes(int lane);
extern vector<double> scanCars(vector<vector<double> > &cars, int prev_size, double min_s, double max_s, double min_d, double max_d);
#endif