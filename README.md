# Dual_pendulum
Control system project

This project was created as a part of an acadamic project for course ENPM 667 - Control system for robotics.

## Dependencies
You must have python3 installed on your machine to run these codes. Following are the required modules,

* numpy
* scipy
* control
* matplotlib

You can install all the dependencies with pip,
```
pip3 install numpy scipy control matplotlib
```

## There are main 3 sections to the project.

### LQR (Linear Quadratic Regulator)

![demo](https://github.com/SaumilShah66/Dual_pendulum/blob/master/demo.gif)

You can change the initial conditions in the file lqr.py and you can also play aroud with Q and R values in the code to get the different results as per need. Though given values are optimized in terms of control and effort to control. Use following command to run the LQR controller,
```
python3 lqr.py
```

### Luenberger Observer

These observer estimates the state of the system taking input from the sensors. observer.py files is not meant to control the system, but estimates the state and will show the animation using estimated data.
```
python3 observer.py
```

### LQG controller

lqg.py file uses state estimator to observe the full state from the sensor data and uses LQR to do the optimum control of the system. In the code, sensor noise and system disturbances are also added with m will try to optimize the non linear system with the nosiy data.
```
python3 lqg.py
```


For more detailed information you can view this [report](https://github.com/SaumilShah66/Dual_pendulum/blob/master/ENPM667_FinalProject_Saumil_Smriti.pdf).
