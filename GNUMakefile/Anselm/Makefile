# /**
# * @File        Makefile
# * @Author      Jiri Jaros
# * @Affiliation FIT, Brno University of Technology
# * @Email       jarosjir@fit.vutbr.cz
# * @Comments    Linux makefile for Anselm
# * 
# * @Tool        kspaceFirstOrder3D 2.15
# * @Created     24 September 2012, 10:57 AM
# * @LastModif   11 February 2014, 17:55 PM
#
# * @License: 
# * This file is part of the C++ extension of the k-Wave Toolbox (http://www.k-wave.org)
# * Copyright (C) 2013 Jiri Jaros and Bradley Treeby
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
#	by GNU g++ 4.3 and newer OR Intel Compiler icpc 11			#
#	The code uses STATIC Linking by default					#
#										#
#	Necessary libraries:							#
#		- FFTW 3.0 and newer OR Intel MKL 11 and newer			#
#		- HDF5 version 1.8 and newer					#
#										#
#	How to compile libraries						#
#		- FFTW : download from "http://www.fftw.org/" 			#
#			run configure script with following parameters:		#
#			--enable-float --enable-sse --enable-openmp		#
#		- MKL  : Only if not using FFTW. Dowload from 			#
#			 "http://software.intel.com/en-us/intel-mkl" 	        #
#		- HDF5 : download from http://www.hdfgroup.org/HDF5/ 		#
#			run configure script with these parameters:	        #		
#			--enable-hl 						#
#									      	#
#	This makefile enables static and dynamic compilation using 		#
#		1) GNU gcc + FFTW  						#
#		2) Intel compiler + Intel MKL					#
#									      	#
#									      	#
#################################################################################


#################################################################################
#	  Set following flags based on your compiler and library paths 		#
#################################################################################

# Select compiler

#COMPILER = Intel

# static lining is deafult
LINKING = STATIC
#LINKING = DYNAMIC


#load following modues for GNU
# module load gcc/4.8.1
# module load hdf5/1.8.13
# module load cuda/6.5.14


#load following modules for Intel
#module load intel/13.5.192(default)
#module load hdf5/1.8.13

#HDF5=$(HOME)/Software/hdf5-1.8.10-serial-icc


#Get GIT hash (only if you biuld form Gitlab repository)
KWAVE_GIT_HASH=$(shell git rev-parse HEAD)
#Otherwise, use the last official build hash

############################## GNU g++ + FFTW ###################################
#GNU version
CXX	   = nvcc

CXXFLAGS = -Xcompiler="-mtune=native -march=native -mavx -O3 -fopenmp -ffast-math -fassociative-math " \
           -std=c++11  -I$(HDF5_INC_DIR)  -I. \
	   -D__KWAVE_GIT_HASH__=\"$(KWAVE_GIT_HASH)\"
  
LDFLAGS  = -Xcompiler="-fopenmp" -Xlinker="-rpath,$(HDF5_LIB_DIR):$(CUDADIR)/lib64" -std=c++11  -L$(HDF5_LIB_DIR)
LIBS     = -lhdf5 -lhdf5_hl -lz -lcufft






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
		MatrixClasses/OutputHDF5Stream/BaseOutputHDF5Stream.o \
		MatrixClasses/OutputHDF5Stream/IndexOutputHDF5Stream.o \
		MatrixClasses/OutputHDF5Stream/WholeDomainOutputHDF5Stream.o \
		MatrixClasses/RealMatrix.o \
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
		MatrixClasses/OutputHDF5Stream/BaseOutputHDF5Stream.o \
		MatrixClasses/OutputHDF5Stream/IndexOutputHDF5Stream.o \
		MatrixClasses/OutputHDF5Stream/WholeDomainOutputHDF5Stream.o \
		MatrixClasses/RealMatrix.o \
		Parameters/CommandLineParameters.o \
		Parameters/Parameters.o \
		$(LIBS)			\
		-o $@


$(TARGET).o : $(TARGET).cpp 
	$(CXX) $(CXXFLAGS) -c $(TARGET).cpp 


CUDA/CUDAImplementations.o : CUDA/CUDAImplementations.cu
	$(CXX) $(CXXFLAGS) -c CUDA/CUDAImplementations.cu -o CUDA/CUDAImplementations.o
clean:
	rm -f *.o HDF5/*.o KSpaceSolver/*.o MatrixClasses/*.o Parameters/*.o Containers/*.o  CUDA/*.o MatrixClasses/OutputHDF5Stream/*.o $(TARGET)


