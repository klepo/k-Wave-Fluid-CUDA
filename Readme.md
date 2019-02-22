## Overview

k-Wave is an open source MATLAB toolbox designed for the time-domain simulation
of propagating acoustic waves in 1D, 2D, or 3D. The toolbox has a wide range of 
functionality, but at its heart is an advanced numerical model that can account 
for both linear or nonlinear wave propagation, an arbitrary distribution of 
weakly heterogeneous material parameters, and power law acoustic  absorption, 
see the http://www.k-wave.org.

This project is a part of the k-Wave toolbox accelerating 3D simulations using 
an optimized CUDA/C++ implementation to run small to moderate grid sizes
(64x64x64 to 512x512x512). This code uses a single NVIDIA GPU to accelerate 
the simulations (AMD GPUs are not supported).

## Repository structure

    .
    +--Containers        - Matrix and output stream containers
    +--Data              - Small test data
    +--GetoptWin64       - Windows version of the getopt routine
    +--Hdf5              - HDF5 classes (file access)
    +--KSpaceSolver      - Solver classes with all the kernels
    +--Logger            - Logger class to report progress and errors
    +--MatrixClasses     - Matrix classes to hold data
    +--Makefiles         - GNU makefiles for different systems
    +--OutputHDF5Streams - Output streams to sample data
    +--Parameters        - Parameters of the simulation
    +--Utils             - Utility routines
    +--nbproject         - NetBeans IDE 8.2 project
    Changelog.md         - Changelog
    License.md           - License file
    Makefile             - NetBeans 8.2 makefile
    Readme.md            - Readme
    Doxyfile             - Doxygen generator file
    header_bg.png        - Doxygen logo
    main.cpp             - Main file of the project


## Compilation
 
The source codes of `kpsaceFirstOrder3D-CUDA` are written using the C++11 
standard (optional OpenMP 2.0  library), NVIDIA CUDA 10.0 library and HDF5 1.8.x. 
 
There are variety of different C++ compilers that can be used to compile the 
source codes. We recommend using the GNU C++ compiler (g++) version 7.3, the
Intel C++ compiler version 2018, or Visual Studio 2017. The version of the 
compiler is limited by the CUDA architecture version (CUDA 10.0 supports GCC up 
to 7.3). The code was tested with CUDA 10.0, however, the code was also tested 
with CUDA 8.0 and 9.x. The codes can be compiled on 64-bit Linux and Windows. 
32-bit systems are not supported due to the the memory requirements even for 
small simulations.
 
 Before compiling the code, it is necessary to install CUDA, C++ compiler and 
 the HDF5 library. The GNU compiler is usually part of Linux distributions and 
 distributed as open source. It can be downloaded from (http://gcc.gnu.org/) if 
 necessary. The Intel compiler can be downloaded from Intel website 
 (https://software.intel.com/en-us/intel-parallel-studio-xe). The Intel compiler
 is only free for non-commercial use.

The CUDA library can be downloaded from the
(https://developer.nvidia.com/cuda-toolkit-archive).
The only supported version is 10.0, however, the code is supposed to  work with 
upcoming CUDA 11.0, but we cannot guarantee it.
 
### The HDF5 library installation procedure

 1. Download the 64-bit HDF5 library 
 https://support.hdfgroup.org/HDF5/release/obtain518.html. Please use version 
 1.8.x, the version 1.10.x is not compatible with MATLAB yet.
  
 2. Configure the HDF5 distribution. Enable the high-level library and specify 
 an installation folder by typing:
    ```bash
    ./configure --enable-hl --prefix=folder_to_install
    ```
 3. Make the HDF library by typing:
    ```bash
    make -j
    ```
 4. Install the HDF5 library by typing:
    ```bash
    make install
    ```
 
### The CUDA installation procedure 
 
  1. Download CUDA version 10.0 
     (https://developer.nvidia.com/cuda-toolkit-archive).
  2. Follow the NVIDIA official installation guide for Windows 
(http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html) 
and Linux (http://docs.nvidia.com/cuda/cuda-installation-guide-linux/).
 
 
### Compiling the CUDA/C++ code 
 
When the libraries and the compiler have been installed, you are ready to 
compile thekspaceFirstOrder3D-CUDA code. The Makefile only supports code 
compilation under CUDA/g++ compiler, however, using different compilers would be
analogous `-ccbin` parameter for the  `nvcc` compiler). 
 
  1. Select the most appropriate makefile. 
     We recommend `Makefiles/Release/Makefile`
  2. Open selected makefile. 
     First, set the paths to CUDA and HDF5 libraries, then select how to link 
     the code. Dynamic lining is preferred since it does not require reloading 
     the CUDA-GPU driver, however, the code will very likely only run on the 
     machine where compiled. The static linking allows to create a 
     self-consistent binary with all libraries linked in. Unfortunately, since 
     there is a bug in the static `cufft` library, it is still necessary to link
     dynamically to this library (and copy the lib file with the binary).
  3. Compile the source code by typing:
    
    ```bash
    make -f Makefiles/Release/Makefile -j 
    ```

Alternatively, the code can be compiled using Netbeans IDE and attached project.

## Usage

The CUDA codes offers a lot of parameters and output flags to be used. For more 
information, please type:

```bash
./kspaceFirstOrder3D-CUDA --help
```
