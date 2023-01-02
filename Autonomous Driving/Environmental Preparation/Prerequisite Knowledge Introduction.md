Ⅰ. Introduction of background knowledge
0x00 Introduction: What is autonomous driving?
" Autonomous (automatic) + driving (driving) "

Autonomous driving, i.e., the technology in which a car autonomously perceives its surroundings and drives safely.

Self-driving car, autonomous vehicle, driver-less car, or robotic car......

0x01 Basic components of autonomous driving
Car: the actual moving vehicle, the agent should control it
Sensors: devices that detect the surrounding environment
Agent: an object that safely drives var in a given surrounding environment


Sensors (Sensors on self-driving cars).

460 LiDAR cameras, RGB cameras


Goal of Autonomous Driving (Goal).

Safe driving in a given situation
Safe driving of the car according to the given situation (state)

Mapping function: Sensor Input → Action

Modular Pipelines (modular Pipelines)
End-to-end training (End-to-End Learning)
Direct Perception (Direct Perception)
0x02 Modular Components (Modular Pipeline)

Each module connects the inputs of the next module: low-level perception, scene resolution, path training, and vehicle control.


❓ Questions to think about: How many degrees should we turn the handle in order to follow the selected path? At what speed should we move forward?


Ⅱ. Pre-requisite knowledge
0x00 Lane marking & Lane detection (Lane marking & Lane detection)
Using gradient maps or edge filtered images, we can detect lane markings by thresholding.
Consider the points where opposite gradients exist in the vicinity.


0x01 Edge detection (Edge Detection)
The gradient map in both directions is obtained by convolving the image with an edge filter.
Other edge kernels can also be used here for edge detection.   


Edge detection is a common technique in image processing for detecting edges and borders in images. This is very important for autonomous driving systems because edge detection helps the system to identify important objects such as roads, vehicles, and pedestrians.

Usually edge detection is achieved by convolving the image using an edge filter, which is a special convolution kernel that contains a gradient map in two directions to detect vertical and horizontal edges in the image. For example, the Sobel filter shown in the figure gives a gradient map in the direction of the image and a gradient map in the direction of the image, where the gradient map in the direction of the image detects the vertical edges and the gradient map in the direction of the image detects the horizontal edges.

In addition to the Sobel filter, there are many other edge kernels that can be used for edge detection. For example, the Canny edge detection algorithm, which is a very popular edge detection algorithm, can effectively eliminate noise and provide clear edge detection results.



Image Gaussian filtering: Blurring the image using a Gaussian filter to reduce noise and make edges more visible.
Calculate Image Gradient: Calculate the gradient of an image using a Sobel filter or other method and use the direction and magnitude of the gradient to represent the edges in the image.
Non-maximum suppression: Use a non-maximum suppression algorithm to remove false edges from an image.
Double Threshold Detection: Uses two thresholds to distinguish between true edges and false edges.
Edge join: Join the detected edges to form a complete edge.

0x02 IPM Inverse Perspective Mapping (IPM)
IPM (Inverse Perspective Mapping) is an image processing technique that inverts a perspective-transformed image to make it look like it was taken from a top-down perspective. This is very important for autonomous driving systems because it helps the system identify objects such as roads, vehicles, and pedestrians more accurately.

In the usual case, the road is on a plane.
If the 3D transformation is known, we can project the road image onto the ground plane.


0x03 Lane line detection: Parametric Lane Marking Estimation (PLME)
In order to navigate the car, we need to match the detected marker pixels with a more semantic curve model with them.



0x04 Bezier Curve
A Bezier curve is a mathematical curve. Bezier curves are commonly used in computer graphics because they can be used to create smooth curves and graphs. A Bezier curve describes the shape of a curve by means of control points, where one or more control points are used to specify the shape of the curve.

Bezier curves can control the shape of a curve by the position of the control points and can change the shape of the curve by changing the position of the control points. This makes Bessel curves very suitable for creating complex curves and shapes.

A polynomial curve defined by control points


0x05 Linear Bezier Curve
A Linear Bezier Curve is a special type of Bezier curve that consists of two control points and a start point and an end point. Linear Bezier Curve is one of the simplest Bezier curves that can be used to describe a straight line. The equation of a linear Bessel curve.


where is the point on the Bessel curve, is the parameter, and and is the control point.

Similar to linear interpolation




0x06 Quadratic Bezier Curve
consists of a start point, an end point and two control points. The Quadratic Bezier Curve is a quadratic equation that can be used to describe curves and complex shapes. The equation of the quadratic Bezier curve is



where is the point on the Bessel curve, is the parameter, and is the control point.

The interpolation of two linear interpolation points




0x07 Cubic Bezier Curve
consists of a start point, a termination point and three control points. A cubic Bezier curve is a cubic equation that can be used to describe curves and complex shapes. The equation of a cubic Bezier curve is



Where, is the point on the Bessel curve, is the parameter, and is the control point.

Interpolation of quadratic points



 0x08 B-Spline Curve (B-Spline Curve)
A B-Spline curve is described by a series of control points that can be used to specify the shape of the curve.B-Spline curves are usually described by the B-Spline curve equation, which is a polynomial equation.There are many different types of B-Spline curves, including quadratic B-Spline curves, cubic B-Spline curves, and quadratic B-Spline curves.

The expression of the quadratic B-sample curve can be defined by knowing the control points


A curve defined by the list of control points and the degree, a single piece polynomial curve (which differs from with a Bessel curve).
