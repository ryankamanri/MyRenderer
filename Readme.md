In short, it is:


# A soft renderer with zero external dependencies for rendering pipeline learning


https://github.com/user-attachments/assets/bb2a456e-94e3-41b8-b1f9-3ee90893346f


Following [Tiny Renderer or how OpenGL works: software rendering in 500 lines of code](https://github.com/ssloy/tinyrenderer), my renderer is designed to demonstrate how the graphics rendering pipeline works, so I won't introduce any external dependencies (including the linear algebra library *glm*, the graphics interface library *glfw*, etc.), just use [WinGDI](https://learn.microsoft.com/en-us/windows/win32/api/wingdi/) to create windows on the Windows platform for real-time rendering.

## How to run
On Windows platform, just execute
``` bash
cd MyRenderer
./sbin/cmake_build.bat
./sbin/ms_build.bat
./sbin/run.bat
```
Other platforms are NOT supported now.


## Remark
I had almost completed all works of the first stage(Rasterization) when I first wrote this `Readme.md`... So excuse me for some case I may forget.  

However, I would try my best to restore the whole procession of how I implemented it and clarify the function of my entire code step by step.  

This `Readme.md` is just a catalog of my introduction docs, and the rest docs is as follows:  

- The essence of my renderer
- The code design and implementation of functions
- The logs in developing

The project is still developing.


=============================================================


**Update on Oct 27, 2023**


The work has almost done because I finished my undergraduate life(as a graduation project).
If anyone is interested in it, see [PDF](docs/2223_51_10636_080901_2019110111_BS.pdf).
