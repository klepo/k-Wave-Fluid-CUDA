# /**
# * @File        Makefile
# * @Author      Jiri Jaros
# *              Faculty of Information Technology
# *              Brno University of Technology 
# * @Email       jarosjir@fit.vutbr.cz
# * @Comments    Linux makefile for Ubuntu 14.04
# * 
# * @Tool        kspaceFirstOrder3D 3.4
# * @Created     02 December  2014, 12:32 
# * @LastModif   18 December  2014, 14:09
#
# * @License: 
# * This file is part of the C++ extension of the k-Wave Toolbox
# * (http://www.k-wave.org).\n Copyright (C) 2014 Jiri Jaros, Beau Johnston
# * and Bradley Treeby
# * 
# * This file is part of the k-Wave. k-Wave is free software: you can redistribute it 
# * and/or modify it under the terms of the GNU Lesser General Public License as 
# * published by the Free Software Foundation, either version 3 of the License, 
# * or (at your option) any later version.
# * 
# * k-Wave is distributed in the hope that it will be useful, but 
# * WITHOUT ANY WARRANTY; without even the implied warranty of 
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  
# * See the GNU Lesser General Public License for more details. 
# * 
# * You should have received a copy of the GNU Lesser General Public License 
# * along with k-Wave. If not, see <http://www.gnu.org/licenses/>.
# */

#################################################################################
#	The source codes can be compiled ONLY under Linux x64 			#
#	by GNU g++ 4.8 and newer OR Intel Compiler icpc 11 and newer 		#
#	The newer compiler, the more advanced instruction set can be used	#
#	We recomend compilation with g++ 4.8/4.9 or icpc 13/14                  #
#                                                                               #
#       The code also requires CUDA 6.5                                         #
#                                                                               #
#	The code requires compilator suporitng C++-11 standard!			#
#	The code uses GNU compiler and STATIC linking by default		#
#										#
#	Necessary libraries:							#
#		- HDF5 version 1.8 and newer					#
#										#
#	How to compile libraries						#
#		- CUDA : download from                                          #
#                           "https://developer.nvidia.com/downloads             #
#		- HDF5 : download from http://www.hdfgroup.org/HDF5/ 		#
#			run configure script with these parameters:	        #		
#			--enable-hl --enable-static --enable-shared     	#
#									      	#
#	This makefile enables static and dynamic compilation using 		#
#									      	#
#									      	#
#################################################################################


#################################################################################
#	  Set following flags based on your compiler and library paths 		#
#################################################################################


# static lining is deafult
LINKING = STATIC
#LINKING = DYNAMIC

#Set up paths: find HDF5
#HDF5_DIR=/usr/local/hdf5-1.8.10-serial-icc
HDF5_DIR=/usr/local/hdf5-serial
CUDA_DIR=/usr/local/cuda

# Select CPU architecture (what instruction set to be used). 
# The libraries such as FFTW, HDF5 and MKL are to be compiled under the same architecture
# e.g. if you want to use AVX in K-Wave, compile FFTW with --enable-avx
CPU_ARCH = native
#CPU_ARCH = SSE3
#CPU_ARCH = SSE4
#CPU_ARCH = AVX
#CPU_ARCH = AVX2

#Get GIT hash (only if you biuld form Gitlab repository)
KWAVE_GIT_HASH=$(shell git rev-parse HEAD)
#Otherwise, use the last official build hash
#KWAVE_GIT_HASH=43e7bcfd05ffa7497ee6a895e34df9a424f56a9e
#################################################################################


############################## NVCC + GNU g++ ###################################
CXX = nvcc
 
# Set compiler flags and header files directories
CXXFLAGS = -Xcompiler="-mtune=native -march=native -O3 -fopenmp -ffast-math -fassociative-math" \
           -std=c++11  -I$(HDF5_DIR)/include  -I. -Xptxas -v   -ccbin=g++-4.8 -arch=compute_52 -code=sm_52
  
LDFLAGS  = -Xcompiler="-fopenmp" -Xlinker="-rpath,$(HDF5_DIR)/lib:$(CUDA_DIR)/lib64" -std=c++11  -L$(HDF5_DIR)/lib  -L$(CUDA_DIR)/lib64 
#LIBS     = -lhdf5 -lhdf5_hl -lz -lcufft
LIBS	  = $(HDF5_DIR)/lib/libhdf5_hl.a 	\
            $(HDF5_DIR)/lib/libhdf5.a		\
            -lz -lcufft

#LIBS	  = $(HDF5_DIR)/lib/libhdf5_hl.a 	\
#            $(HDF5_DIR)/lib/libhdf5.a		\
#            $(CUDA_DIR)/lib64/libcufft_static.a \
#            $(CUDA_DIR)/lib64/libcufftw_static.a \
#	    -lz -ldl

  

################################# Compile #####################################

TARGET		= kspaceFirstOrder3D-CUDA

all:		$(TARGET)	


$(TARGET):	CUDA/CUDAImplementations.o \
		CUDA/CUDATuner.o \
		Containers/MatrixContainer.o \
		Containers/MatrixRecord.o \
		Containers/OutputStreamContainer.o \
		HDF5/HDF5_File.o \
		KSpaceSolver/KSpaceFirstOrder3DSolver.o \
		MatrixClasses/BaseFloatMatrix.o \
		MatrixClasses/BaseIndexMatrix.o \
		MatrixClasses/CUFFTComplexMatrix.o \
		MatrixClasses/ComplexMatrix.o \
		MatrixClasses/IndexMatrix.o \
		MatrixClasses/RealMatrix.o \
		OutputHDF5Streams/BaseOutputHDF5Stream.o \
		OutputHDF5Streams/IndexOutputHDF5Stream.o \
		OutputHDF5Streams/CuboidOutputHDF5Stream.o \
		OutputHDF5Streams/WholeDomainOutputHDF5Stream.o \
		OutputHDF5Streams/OutputStreamsCUDAKernels.o \
		Parameters/CommandLineParameters.o \
		Parameters/Parameters.o \
		main.o


	$(CXX) $(LDFLAGS) main.o	\
		CUDA/CUDAImplementations.o \
		CUDA/CUDATuner.o \
		Containers/MatrixContainer.o \
		Containers/MatrixRecord.o \
		Containers/OutputStreamContainer.o \
		HDF5/HDF5_File.o \
		KSpaceSolver/KSpaceFirstOrder3DSolver.o \
		MatrixClasses/BaseFloatMatrix.o \
		MatrixClasses/BaseIndexMatrix.o \
		MatrixClasses/CUFFTComplexMatrix.o \
		MatrixClasses/ComplexMatrix.o \
		MatrixClasses/IndexMatrix.o \
		MatrixClasses/RealMatrix.o \
		OutputHDF5Streams/BaseOutputHDF5Stream.o \
		OutputHDF5Streams/IndexOutputHDF5Stream.o \
		OutputHDF5Streams/CuboidOutputHDF5Stream.o \
		OutputHDF5Streams/WholeDomainOutputHDF5Stream.o \
		OutputHDF5Streams/OutputStreamsCUDAKernels.o \
		Parameters/CommandLineParameters.o \
		Parameters/Parameters.o \
		$(LIBS)			\
		-o $@

$(TARGET).o : $(TARGET).cpp 
	$(CXX) $(CXXFLAGS) -c $(TARGET).cpp 


CUDA/CUDAImplementations.o : CUDA/CUDAImplementations.cu
	$(CXX) $(CXXFLAGS) -c CUDA/CUDAImplementations.cu -o CUDA/CUDAImplementations.o
	
OutputHDF5Streams/OutputStreamsCUDAKernels.o : OutputHDF5Streams/OutputStreamsCUDAKernels.cu
	$(CXX) $(CXXFLAGS) -c OutputHDF5Streams/OutputStreamsCUDAKernels.cu -o OutputHDF5Streams/OutputStreamsCUDAKernels.o
	
clean:
	rm -f *.o HDF5/*.o KSpaceSolver/*.o MatrixClasses/*.o Parameters/*.o Containers/*.o  CUDA/*.o OutputHDF5Streams/*.o $(TARGET)



