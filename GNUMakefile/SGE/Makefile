# /**
# * @File        Makefile
# * @Author      Jiri Jaros
# *              Faculty of Information Technology
# *              Brno University of Technology 
# * @Email       jarosjir@fit.vutbr.cz
# * @Comments    Linux makefile for BUT SGE cluster
# * 
# * @Tool        kspaceFirstOrder3D 3.4
# * @Created     02 December  2014, 12:32 
# * @LastModif   14 July      2016, 16:54
#
# * @License: 
# * This file is part of the C++ extension of the k-Wave Toolbox
# * (http://www.k-wave.org).\n Copyright (C) 2016 Jiri Jaros, Beau Johnston
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
#   This makefile is a tailored for the SGE supercomputer                       #
#   Please load following modules:                                              #
#      gcc/4.8.1 cuda/7.5 hdf5/1.8.13                                           #
#                                                                               #
#   The code requires compilator suporitng C++-11 standard!                     #
#                                                                               #
#   This makefile enables dynamic compilation only                              #
#                                                                               #
#################################################################################


#################################################################################
#        Set following flags based on your compiler and library paths           #
#################################################################################

# static lining is deafult
LINKING = STATIC
#LINKING = DYNAMIC

#Set up paths: find HDF5
#HDF5_DIR=/usr/local/hdf5-1.8.10-serial-icc
HDF5_DIR=/usr/local/lib
CUDA_DIR=/usr/local/share/cuda-7.5.18

#Get GIT hash (only if you biuld form Gitlab repository)
KWAVE_GIT_HASH=$(shell git rev-parse HEAD)
#Otherwise, use the last official build hash
#KWAVE_GIT_HASH=43e7bcfd05ffa7497ee6a895e34df9a424f56a9e
#################################################################################


############################## NVCC + GNU g++ ###################################
CXX = $(CUDA_DIR)/bin/nvcc
 
# Set compiler flags and header files directories
CXXFLAGS = -Xcompiler="-Wall -O3 -fopenmp -ffast-math -fassociative-math -march=native" \
           -O3 -std=c++11  -I$(HDF5_DIR)/include  -I.  -ccbin=g++-4.8 --restrict\
	    -D__KWAVE_GIT_HASH__=\"$(KWAVE_GIT_HASH)\"  \
	    --generate-code arch=compute_30,code=sm_30	\
	    --generate-code arch=compute_32,code=sm_32	\
	    --generate-code arch=compute_35,code=sm_35	\
	    --generate-code arch=compute_37,code=sm_37	\
	    --generate-code arch=compute_50,code=sm_50	\
	    --generate-code arch=compute_52,code=sm_52	\
	    --generate-code arch=compute_53,code=sm_53	\
	    --device-c

LDFLAGS  = -Xcompiler="-fopenmp" -Xlinker="-rpath,$(HDF5_DIR)/lib:$(CUDA_DIR)/lib64" -std=c++11  -L$(HDF5_DIR)/lib  -L$(CUDA_DIR)/lib64 -ccbin=g++-4.8
LIBS     = -lhdf5 -lhdf5_hl -lz -lcufft



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
	rm -f *.o HDF5/*.o KSpaceSolver/*.o MatrixClasses/*.o Parameters/*.o Containers/*.o  OutputHDF5Streams/*.o Logger/*.o $(TARGET)

