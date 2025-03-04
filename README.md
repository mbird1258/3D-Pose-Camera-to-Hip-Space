# Transforming Body Pose Coordinates From Hip Space to Camera Space
<img width="350" alt="Geogebra Export" src="https://github.com/user-attachments/assets/b48b1960-715f-4b3c-ad0c-e1fb7c7e33b2">

Blog post: [https://matthew-bird.com/blogs/Hip-to-Camera-Space.html](https://matthew-bird.com/blogs/Hip-to-Camera-Space.html)

## Premise
I’ve always been somewhat annoyed that many of the off-the-shelf 3D pose models, such as [mediapipe](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker), return 3D pose coordinates that are centered around the hips, as I feel that they become almost impossible to build off of since camera space coordinates are more useful in 99% of circumstances. Thus, my plan for this project was to create an algorithm that finds the translation in 3D space that moves a body from hip space into camera space. 

## Instructions for Setup
Download pose landmarker [here](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python) and rename to pose_landmarker.task 

Download model [here](https://bit.ly/metrabs_l) and rename folder to model

## Fitting 3D Coordinates to set of 3D Lines
Our end goal is to have our body’s pose represented in 3D space with the origin of 3D space being our camera. Thus, we can place our camera at (0,0,0), with +x direction representing right, +y representing up, and +z representing forward. Using this, we can plot our 2D pose coordinates on our screen in 3D on the camera’s image plane.  

<img width="350" alt="Geogebra Export" src="https://github.com/user-attachments/assets/d4ced79e-621d-4be6-a354-5341f4a63dfd">

Since we know that our 2D pose is the projection of the 3D pose onto the image plane, each point in our 3D pose must lie on the line starting from (0,0,0) (the camera’s position) and going through the corresponding 2D pose point on the image plane. 

<img width="350" alt="Geogebra Export" src="https://github.com/user-attachments/assets/b48b1960-715f-4b3c-ad0c-e1fb7c7e33b2">

By using the offset between 2 joints in our 3D pose, (dx2,dy2,dz2), we can fit the line segment that represents this to our lines starting from (0,0,0) and passing through the coordinates on the image plane representing our first and second joints, (x1,y1,z) and (x1',y1',z'). We do this by projecting our line segment and 2 lines onto the xz plane and doing the following operations:

<img width="350" alt="2169" src="https://github.com/user-attachments/assets/9cf3464e-618c-416c-bdc0-f53d5decb0bc">

After this, we can do the same for the projection onto the yz plane, giving us our final (x2,y2,z2) and (x2',y2',z2') representing the 3D positions of the joints around the camera. The translation to get from (x1,y1,z) to (x2,y2,z) given by (x2-x1,y2-y1,z2-z1) can then be used to translate all of our 3D pose’s points from hip space to camera space. The final results of alignment can be seen [here](https://www.geogebra.org/calculator/rz5tpm7b). 

## Calibrating Camera View Depth
The above derivation of the translation to move from hip space to camera space requires a value, z, that we do not have yet, which represents the distance from the camera to the image plane. 

### Attempt 1
In the first attempt, I made use of the fact that projecting onto the xz plane gave numbers for both x2 and z2, and projecting onto the yz plane gave numbers for both y2 and z2. Thus, we could equate the two formulas for z2 and solve for z. The geogebra implementation of this is linked [here](https://www.geogebra.org/calculator/mgqds25s), but after implementing it, I found that it was way too error prone and would return essentially completely random values of z. 

<img width="350" alt="78833" src="https://github.com/user-attachments/assets/3415922a-ed6c-41cb-ba49-5e7e7852ed48">

### Attempt 2
In the second attempt, I used scipy’s optimize module to find the value for z that would minimize the error of the fitted 3D pose to the lines. However, I once again found that it was way too error prone and returned highly inaccurate values. 

### Attempt 3
In the third attempt, I made use of [this 3D pose model](https://istvansarandi.com/eccv22_demo/)(that took me ages of digging to find) to find the 2D and 3D pose of the bodies in a scene. This differs from other models like mediapipe’s as it directly returns the 3D pose in camera space, thus allowing me to find the value of z that would minimize the distance of the 3D pose from the lines starting at (0,0,0) and going through the 2D pose on the image plane. The drawbacks to this model are that it seems to take ages to load and run, and thus it cannot really be used to find the 3D pose in camera space for every frame, but for our purposes this is perfect as we only need to run it once for a certain model of camera to find the distance between the camera’s origin and its image plane, since this distance should remain constant across any video. 

## Conclusion
I’m extremely happy with how this project turned out. I managed to achieve two long standing goals for myself: getting 3D pose positions in camera space, and finding the distance between a camera and its image plane automatically - issues that I was facing for quite some time now but were seemingly overlooked when they seemed like such fundamental problems to solve. 
