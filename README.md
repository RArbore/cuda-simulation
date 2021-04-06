![alt text](https://github.com/RArbore/cuda-simulation/blob/main/1.png?raw=true)
![alt text](https://github.com/RArbore/cuda-simulation/blob/main/2.png?raw=true)
![alt text](https://github.com/RArbore/cuda-simulation/blob/main/3.png?raw=true)

To compile with NVCC, use these flags: -lGL -lGLU -lglut --expt-relaxed-constexpr

If you would like to run the compiled binary without VSync locking your FPS, you can use "__GL_SYNC_TO_VBLANK=0". Note, to run a GL application with this option, you must type this option before the path to the binary. In the case of glxgears, for example, you would run "__GL_SYNC_TO_VBLANK=0 glxgears".

This simulation is based on: https://www.youtube.com/watch?v=X-iSQQgOd1A
