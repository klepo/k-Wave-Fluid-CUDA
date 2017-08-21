# /**
# * @file        Makefile
# * @author      Jiri Jaros
# *              Faculty of Information Technology
# *              Brno University of Technology 
# * @email       jarosjir@fit.vutbr.cz
# * @comments    Linux makefile for Anselm supercomputer
# * 
# * @tool        kspaceFirstOrder3D 3.5
# * @created     02 December  2014, 12:32 
# * @kastModif   21 August    2017, 14:23
# *
# * @copyright Copyright (C) 2017 Jiri Jaros and Bradley Treeby.
#  *
#  * This file is part of the C++ extension of the k-Wave Toolbox 
#  * [k-Wave Toolbox](http://www.k-wave.org). 
#  *
#  * This file is part of the k-Wave. k-Wave is free software: you can 
#  * redistribute it and/or modify it under the terms of the GNU Lesser General 
#  * Public License as published by the Free Software Foundation, either version 
#  * 3 of the License, or (at your option) any later version.
#  *
#  * k-Wave is distributed in the hope that it will be useful, but WITHOUT ANY 
#  * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
#  * FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for
#  * more details.
#  *
#  * You should have received a copy of the GNU Lesser General Public License 
#  * along with k-Wave.
#  * If not, see [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/).
#  */

################################################################################
# The source codes can be compiled ONLY under Linux x64 by GNU g++ 4.9 and     #
# newer or Intel Compiler icpc 15 and newer. The newer compiler, the more      #
# advanced instruction set can be used.                                        #
# We recommend compilation with g++ 5.3/5.4 or icpc 16                         #
#                                                                              #
# The code also requires CUDA 8.0, but can work with CUDA 9.0 RC1              #
# The code cannot be compiled with older versions of CUDA since                #
# the incompatibility with the cuda FFT library.                               #
#                                                                              #
# This makefile is prepared for the GNU compiler semi-staic linking by default #
# This means all libraries are linked statically but cuFFT due to a bug in     #
# the static cuFFT library, we link against                                    #
# cuda dynamically. This also enables us to reuse loaded cuda driver and do    #
# not force the driver reload.                                                 #
#                                                                              #
# Necessary libraries:                                                         #
#  - HDF5 version 1.8.x                                                        #
#                                                                              #
#  How to compile libraries                                                    #
#  - CUDA : download from                                                      #
#     https://developer.nvidia.com/cuda-toolkit-archive                        #
#  - HDF5 : download from                                                      #
#     https://support.hdfgroup.org/HDF5/release/obtain518.html                 #
#     run configure script with these parameters:                              #
#       --enable-hl --enable-static --enable-shared                            #
#                                                                              #
# How to compile the code                                                      #
#  - Select one of the linking possibilities                                   #
#  - Fill necessary paths                                                      #
#  - make -j                                                                   #
#                                                                              # 
################################################################################


################################################################################
#	  Set following flags based on your compiler and library paths         #
################################################################################


# Semi-static lining is default
# everything will be linked statically, may not work on all GPUs
#LINKING = STATIC
# everything will be linked dynamically
#LINKING = DYNAMIC
# everything but cufft will be linked statically. 
LINKING = SEMI

#Set up paths to necessary libraries: 
HDF5_DIR=/home_lustre/jarosjir/Software/hdf5-serial/1.8.19/gcc5.3.0-sse2
CUDA_DIR=${CUDA_HOME}
ZLIB_DIR=/home/jarosjir/Software/zlib/1.2.8-gcc53-sse2
SZIP_DIR=/home/jarosjir/Software/szip/2.1-gcc53-sse2
    
#Get GIT hash (only if you build form Gitlab repository)
#KWAVE_GIT_HASH=$(shell git rev-parse HEAD)
#Otherwise, use the last official build hash
KWAVE_GIT_HASH=18b205f0e4a37175f067c66bc3e34a731503726d


############################## NVCC + GNU g++ ##################################
CXX = nvcc
 

# Set compiler flags and header files directories
CXXFLAGS = -Xcompiler="-Wall -O3 -fopenmp -ffast-math -fassociative-math" \
           -O3 -std=c++11  -I$(HDF5_DIR)/include  -I.  --restrict \
	   -D__KWAVE_GIT_HASH__=\"$(KWAVE_GIT_HASH)\"   \
   	   --device-c
	   
CUDA_ARCH = --generate-code arch=compute_20,code=sm_20	\
	    --generate-code arch=compute_20,code=sm_21	\
	    --generate-code arch=compute_30,code=sm_30	\
	    --generate-code arch=compute_32,code=sm_32	\
	    --generate-code arch=compute_35,code=sm_35	\
	    --generate-code arch=compute_37,code=sm_37	\
	    --generate-code arch=compute_50,code=sm_50	\
	    --generate-code arch=compute_52,code=sm_52	\
	    --generate-code arch=compute_53,code=sm_53	\
	    --generate-code arch=compute_60,code=sm_60	\
	    --generate-code arch=compute_61,code=sm_61	\


  
LDFLAGS  =  -Xcompiler="-fopenmp" \
	    -Xlinker="-rpath,$(HDF5_DIR)/lib:$(CUDA_DIR)/lib64" -std=c++11  \
	    -L$(HDF5_DIR)/lib  -L$(CUDA_DIR)/lib64 

ifeq ($(LINKING),STATIC)
	LIBS  = $(HDF5_DIR)/lib/libhdf5_hl.a 	     \
            	$(HDF5_DIR)/lib/libhdf5.a	     \
	        $(CUDA_DIR)/lib64/libcufft_static.a  \
	        $(CUDA_DIR)/lib64/libculibos.a       \
        	$(CUDA_DIR)/lib64/libcudart_static.a \
	        $(ZLIB_DIR)/lib/libz.a		     \
	        $(SZIP_DIR)/lib/libsz.a 	     \
		-ldl
endif

ifeq ($(LINKING),DYNAMIC)
	LIBS     = -lhdf5 -lhdf5_hl -lz -lcufft 
endif
  
ifeq ($(LINKING),SEMI)
	LIBS  = $(HDF5_DIR)/lib/libhdf5_hl.a 	\
            	$(HDF5_DIR)/lib/libhdf5.a	\
	        $(ZLIB_DIR)/lib/libz.a		\
	        $(SZIP_DIR)/lib/libsz.a 	\
	        -lcufft                         \
		-ldl
endif

################################# Compile #####################################

TARGET		= kspaceFirstOrder3D-CUDA

all:		$(TARGET)	


$(TARGET):	Containers/MatrixContainer.o             \
		Containers/MatrixRecord.o                \
		Containers/OutputStreamContainer.o       \
		Hdf5/Hdf5File.o                          \
		Hdf5/Hdf5FileHeader.o                    \
		KSpaceSolver/KSpaceFirstOrder3DSolver.o  \
		KSpaceSolver/SolverCudaKernels.o         \
		Logger/Logger.o	                         \
		MatrixClasses/BaseFloatMatrix.o          \
		MatrixClasses/BaseIndexMatrix.o          \
		MatrixClasses/CufftComplexMatrix.o       \
		MatrixClasses/ComplexMatrix.o            \
		MatrixClasses/IndexMatrix.o              \
		MatrixClasses/RealMatrix.o               \
		OutputStreams/BaseOutputStream.o         \
		OutputStreams/IndexOutputStream.o        \
		OutputStreams/CuboidOutputStream.o       \
		OutputStreams/WholeDomainOutputStream.o  \
		OutputStreams/OutputStreamsCudaKernels.o \
		Parameters/CommandLineParameters.o       \
		Parameters/Parameters.o                  \
		Parameters/CudaParameters.o              \
		Parameters/CudaDeviceConstants.o         \
		main.o


	$(CXX) $(LDFLAGS) $(CUDA_ARCH) main.o	         \
		Containers/MatrixContainer.o             \
		Containers/MatrixRecord.o                \
		Containers/OutputStreamContainer.o       \
		Hdf5/Hdf5File.o                          \
		Hdf5/Hdf5FileHeader.o                    \
		KSpaceSolver/KSpaceFirstOrder3DSolver.o  \
		KSpaceSolver/SolverCudaKernels.o         \
		Logger/Logger.o	                         \
		MatrixClasses/BaseFloatMatrix.o          \
		MatrixClasses/BaseIndexMatrix.o          \
		MatrixClasses/CufftComplexMatrix.o       \
		MatrixClasses/ComplexMatrix.o            \
		MatrixClasses/IndexMatrix.o              \
		MatrixClasses/RealMatrix.o               \
		OutputStreams/BaseOutputStream.o         \
		OutputStreams/IndexOutputStream.o        \
		OutputStreams/CuboidOutputStream.o       \
		OutputStreams/WholeDomainOutputStream.o  \
		OutputStreams/OutputStreamsCudaKernels.o \
		Parameters/CommandLineParameters.o       \
		Parameters/Parameters.o                  \
		Parameters/CudaParameters.o              \
		Parameters/CudaDeviceConstants.o         \
		$(LIBS)			                 \
		-o $@

$(TARGET).o : $(TARGET).cpp 
	$(CXX) $(CXXFLAGS) $(CUDA_ARCH) -c $(TARGET).cpp 


KSpaceSolver/SolverCudaKernels.o : KSpaceSolver/SolverCudaKernels.cu
	$(CXX) $(CXXFLAGS) $(CUDA_ARCH) -c KSpaceSolver/SolverCudaKernels.cu -o KSpaceSolver/SolverCudaKernels.o

OutputStreams/OutputStreamsCudaKernels.o : OutputStreams/OutputStreamsCudaKernels.cu
	$(CXX) $(CXXFLAGS) $(CUDA_ARCH) -c OutputStreams/OutputStreamsCudaKernels.cu -o OutputStreams/OutputStreamsCudaKernels.o

Parameters/CudaDeviceConstants.o \ : Parameters/CudaDeviceConstants.cu
	$(CXX) $(CXXFLAGS) $(CUDA_ARCH) -c Parameters/CudaDeviceConstants.cu -o Parameters/CudaDeviceConstants.o

clean:
	rm -f *.o Hdf5/*.o KSpaceSolver/*.o MatrixClasses/*.o Parameters/*.o Containers/*.o  OutputStreams/*.o Logger/*.o $(TARGET)
