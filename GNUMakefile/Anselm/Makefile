# /**
# * @File        Makefile
# * @Author      Jiri Jaros
# *              Faculty of Information Technology
# *              Brno University of Technology 
# * @Email       jarosjir@fit.vutbr.cz
# * @Comments    Linux makefile for Anselm
# * 
# * @Tool        kspaceFirstOrder3D 3.4
# * @Created     02 December  2014, 12:32 
# * @LastModif   20 April     2016, 11:10
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
#HDF5_DIR=/usr/local/hdf5-serial
#CUDA_DIR=/usr/local/cuda

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


HDF5_LIB_DIR=/home/jarosjir/Software/hdf5-serial/1.8.16/gcc4.8.1/lib
HDF5_INC_DIR=/home/jarosjir/Software/hdf5-serial/1.8.16/gcc4.8.1/include
############################## NVCC + GNU g++ ###################################
CXX = nvcc
 
# Set compiler flags and header files directories
CXXFLAGS = -Xcompiler="-Wall -O3 -fopenmp -ffast-math -fassociative-math -mavx -march=native" \
           -O3 -std=c++11  -I$(HDF5_INC_DIR)  -I. --restrict \
	    --generate-code arch=compute_20,code=sm_20	\
	    --generate-code arch=compute_20,code=sm_21	\
	    --generate-code arch=compute_30,code=sm_30	\
	    --generate-code arch=compute_32,code=sm_32	\
	    --generate-code arch=compute_35,code=sm_35	\
	    --generate-code arch=compute_37,code=sm_37	\
	    --generate-code arch=compute_50,code=sm_50	\
	    --generate-code arch=compute_52,code=sm_52	\
	    --generate-code arch=compute_53,code=sm_53	\
	    --device-c


LDFLAGS  = -Xcompiler="-fopenmp " -Xlinker="-rpath,$(HDF5_LIB_DIR):$(CUDADIR)/lib64" -std=c++11  -L$(HDF5_LIB_DIR)/lib  -L$(CUDADIR)/lib64
#LIBS     = -lhdf5 -lhdf5_hl -lz -lcufft 
#LIBS	  = $(HDF5_DIR)/lib/libhdf5_hl.a 	\
#            $(HDF5_DIR)/lib/libhdf5.a		\
#            -lz -lcufft

LIBS	  = $(HDF5_LIB_DIR)/libhdf5_hl.a 	\
            $(HDF5_LIB_DIR)/libhdf5.a		\
            $(CUDADIR)/lib64/libcufft_static.a \
            $(CUDADIR)/lib64/libculibos.a \
            $(CUDADIR)/lib64/libculibos.a \
            $(CUDADIR)/lib64/libcudart_static.a \
            /apps/libs/szip/gcc/2.1/lib/libsz.a	\
            /apps/libs/zlib/gcc/1.2.8/lib/libz.a \
	    -ldl

  

################################# Compile #####################################

TARGET		= kspaceFirstOrder3D-CUDA

all:		$(TARGET)	


$(TARGET):	Containers/MatrixContainer.o \
		Containers/MatrixRecord.o \
		Containers/OutputStreamContainer.o \
		HDF5/HDF5_File.o \
		KSpaceSolver/KSpaceFirstOrder3DSolver.o \
		KSpaceSolver/SolverCUDAKernels.o \
		Logger/Logger.o	\
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
		Parameters/CUDAParameters.o \
		Parameters/CUDADeviceConstants.o \
		main.o


	$(CXX) $(LDFLAGS) main.o	\
		Containers/MatrixContainer.o \
		Containers/MatrixRecord.o \
		Containers/OutputStreamContainer.o \
		HDF5/HDF5_File.o \
		KSpaceSolver/KSpaceFirstOrder3DSolver.o \
		KSpaceSolver/SolverCUDAKernels.o \
		Logger/Logger.o	\
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
		Parameters/CUDAParameters.o \
		Parameters/CUDADeviceConstants.o \
		$(LIBS)			\
		-o $@

$(TARGET).o : $(TARGET).cpp 
	$(CXX) $(CXXFLAGS) -c $(TARGET).cpp 


KSpaceSolver/SolverCUDAKernels.o : KSpaceSolver/SolverCUDAKernels.cu
	$(CXX) $(CXXFLAGS) -c KSpaceSolver/SolverCUDAKernels.cu -o KSpaceSolver/SolverCUDAKernels.o
	
OutputHDF5Streams/OutputStreamsCUDAKernels.o : OutputHDF5Streams/OutputStreamsCUDAKernels.cu
	$(CXX) $(CXXFLAGS) -c OutputHDF5Streams/OutputStreamsCUDAKernels.cu -o OutputHDF5Streams/OutputStreamsCUDAKernels.o

Parameters/CUDADeviceConstants.o \ : Parameters/CUDADeviceConstants.cu
	$(CXX) $(CXXFLAGS) -c Parameters/CUDADeviceConstants.cu -o Parameters/CUDADeviceConstants.o
	
clean:
	rm -f *.o HDF5/*.o KSpaceSolver/*.o MatrixClasses/*.o Parameters/*.o Containers/*.o  OutputHDF5Streams/*.o $(TARGET)



