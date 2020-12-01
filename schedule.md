## Schedule

### Week 1 11/04 - 11/11 

* We researched the topic of SPH more thoroughly exploring different aspects of the algorithm, from how to calculate pressure and densities, to what type of smoothing kernels are typically used.
* Installed CUDA on our main work PCs and ensured compatibility with our GPUs
* Outlined a brute force O(N^2) algorithm for SPH 

### Week 2 11/11 - 11/18
* Started writing some OpenGL to draw 3D objects on screen. 
* Designed Particle datastructure
* We realized that we were spending too much time writing/learning OpenGL so we decided to build our project on top of one of NVIDIA's Particle Demo which already had a graphical solution in OpenGL.
* Stripped out the Particle System out of the NVIDIA particle demo, removed modules that weren't useful to us

### Week 3 11/18 - 11/25 
* Wrote brute force SPH solver ontop of the NVIDIA particle demo
* Lots of debugging/parameter tuning to get particles to move correctly
* More research into how to make our simulation more stable numerically
* Added some features to the particle demo such as increasing/decreasing size of simulation box

### Week 4 11/25 - 12/2
* Thanksgiving Break 
* More debugging to get particle solver to work
* Added Particle Collision
* Added Z-indexing sorting
* Added OpenMP support for bottleneck for loops

### Week 5.0 12/2 - 12/05
* Debug particle collisions using OpenMP - Oscar
* Write GPU kernels for density and pressure calculations - Logan

### Week 5.5 12/05 - 12/09
* Write GPU kernels for force calculations - Oscar
* Write GPU kernels for velocity and position integration - Logan
* Parameter tweaking - Oscar

### Week 6.0 12/09 - 12/12
* Implement GPU kernel for z-index sorting of particles - Logan
* Improve OpenMP load balancing - Oscar
* Testing - Oscar

### Week 6.5 12/12 - 12/15
* Parameter tweaking - Logan
* Data gathering - Oscar
* Create demo scenes - Both
* Prepare final report and presentation - Both
