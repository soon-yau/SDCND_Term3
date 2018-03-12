Code structures:
- ignore planner.cpp and .hpp, they were from earlier experiments
- I moved the helper functions from main.cpp into helper.cpp which also include some of my own functions to simplify main algorithm
- all path planning algorithmic code resides in main.cpp

My algorithm works like this:
- from sensor fusion data, if there are cars in 60 meters ahead in the same lane, then prepare to slow down
- if it is slowing down, then look for adjacent lane to see if there is space for lane change
- if there is space for lane change, set the lane as target and cancel slow down
- the rest of the code is to use generate smooth trajectory and they were inspired by the walkthrough video including the use of spline and calculation of car direction from previous waypoints


