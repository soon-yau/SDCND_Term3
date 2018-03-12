#include<iostream>
#include<vector>

#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "Eigen-3.3/Eigen/LU"
#include "helper.hpp"
#include "planner.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

vector<double> Planner::JMT(CarState start, CarState end, double T)
{
    /*
    Calculate the Jerk Minimizing Trajectory that connects the initial state
    to the final state in time T.

    INPUTS

    start - the vehicles start location given as a length three array
        corresponding to initial values of [s, s_dot, s_double_dot]

    end   - the desired end state for vehicle. Like "start" this is a
        length three array.

    T     - The duration, in seconds, over which this maneuver should occur.

    OUTPUT 
    an array of length 6, each value corresponding to a coefficent in the polynomial 
    s(t) = a_0 + a_1 * t + a_2 * t**2 + a_3 * t**3 + a_4 * t**4 + a_5 * t**5

    EXAMPLE

    > JMT( [0, 10, 0], [10, 10, 0], 1)
    [0.0, 10.0, 0.0, 0.0, 0.0, 0.0]
    */
    vector <double> alpha {start.pos,
                           start.dot,
                           0.5*start.dotdot, 
                           0, 0, 0};
    
    MatrixXd S(3, 1);
    S(0) = end.pos;
    S(1) = end.dot;
    S(2) = end.dotdot;
    
    MatrixXd C(3, 1);    
    C(0) = alpha[0]+alpha[1]*T+alpha[2]*T*T;
    C(1) = alpha[1]+2*alpha[2]*T;
    C(2) = 2*alpha[2];

    double T2=T*T;
    double T3=T2*T;
    double T4=T3*T;
    double T5=T4*T;    
    MatrixXd T_mat(3, 3);
    T_mat << T3, T4, T5,
             3*T2, 4*T3, 5*T4,
             6*T, 12*T2, 20*T3;
    
    MatrixXd alpha345(3, 1);
    alpha345 = T_mat.inverse()*(S-C);
    
    alpha[3]=alpha345(0);
    alpha[4]=alpha345(1);
    alpha[5]=alpha345(2);
    return alpha;
    
}

void Planner::generate(void)
{
    CarState start{0,0,0};
    CarState end{10,0,0};
    double T=5;
    auto coef=JMT(start, end, T);
    for (auto c:coef)
        std::cout<<c<<' ';
}