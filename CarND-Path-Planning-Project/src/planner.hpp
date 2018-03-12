#ifndef PLANNER_HPP
#define PLANNER_HPP

#include <vector>
#define MAX_ACCEL 5
#define MAX_SPEED_MS (49.5*0.44704)
#define TIME_STEP (0.02)
// in Fernet coordinate and metric unit
typedef struct CarState
{
    double pos;
    double dot;
    double dotdot;
}JerkState;

class Planner{
    public:
        Planner(){};
        vector<double> JMT(CarState start, CarState end, double T);
        void generate(void);

        vector<double> next_x;
        vector<double> next_y;        
};

#endif