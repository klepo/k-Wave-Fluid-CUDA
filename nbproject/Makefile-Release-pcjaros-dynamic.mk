#
# Generated Makefile - do not edit!
#
# Edit the Makefile in the project folder instead (../Makefile). Each target
# has a -pre and a -post target defined where you can add customized code.
#
# This makefile implements configuration specific macros and targets.


# Environment
MKDIR=mkdir
CP=cp
GREP=grep
NM=nm
CCADMIN=CCadmin
RANLIB=ranlib
CC=nvcc
CCC=nvcc
CXX=nvcc
FC=gfortran
AS=as

# Macros
CND_PLATFORM=CUDA-Linux
CND_DLIB_EXT=so
CND_CONF=Release-pcjaros-dynamic
CND_DISTDIR=dist
CND_BUILDDIR=build

# Include project Makefile
include Makefile

# Object Directory
OBJECTDIR=${CND_BUILDDIR}/${CND_CONF}/${CND_PLATFORM}

# Object Files
OBJECTFILES= \
	${OBJECTDIR}/Containers/MatrixContainer.o \
	${OBJECTDIR}/Containers/MatrixRecord.o \
	${OBJECTDIR}/Containers/OutputStreamContainer.o \
	${OBJECTDIR}/Hdf5/Hdf5File.o \
	${OBJECTDIR}/Hdf5/Hdf5FileHeader.o \
	${OBJECTDIR}/KSpaceSolver/KSpaceFirstOrder3DSolver.o \
	${OBJECTDIR}/KSpaceSolver/SolverCudaKernels.o \
	${OBJECTDIR}/Logger/Logger.o \
	${OBJECTDIR}/MatrixClasses/BaseFloatMatrix.o \
	${OBJECTDIR}/MatrixClasses/BaseIndexMatrix.o \
	${OBJECTDIR}/MatrixClasses/ComplexMatrix.o \
	${OBJECTDIR}/MatrixClasses/CufftComplexMatrix.o \
	${OBJECTDIR}/MatrixClasses/IndexMatrix.o \
	${OBJECTDIR}/MatrixClasses/RealMatrix.o \
	${OBJECTDIR}/OutputStreams/BaseOutputStream.o \
	${OBJECTDIR}/OutputStreams/CuboidOutputStream.o \
	${OBJECTDIR}/OutputStreams/IndexOutputStream.o \
	${OBJECTDIR}/OutputStreams/OutputStreamsCudaKernels.o \
	${OBJECTDIR}/OutputStreams/WholeDomainOutputStream.o \
	${OBJECTDIR}/Parameters/CommandLineParameters.o \
	${OBJECTDIR}/Parameters/CudaDeviceConstants.o \
	${OBJECTDIR}/Parameters/CudaParameters.o \
	${OBJECTDIR}/Parameters/Parameters.o \
	${OBJECTDIR}/main.o


# C Compiler Flags
CFLAGS=-m64

# CC Compiler Flags
CCFLAGS=-Xcompiler="-O3 -Wall -fopenmp -mavx -march=native -mtune=native" -arch=compute_52 -code=sm_52 --device-c
CXXFLAGS=-Xcompiler="-O3 -Wall -fopenmp -mavx -march=native -mtune=native" -arch=compute_52 -code=sm_52 --device-c

# Fortran Compiler Flags
FFLAGS=

# Assembler Flags
ASFLAGS=

# Link Libraries and Options
LDLIBSOPTIONS=-L${EBROOTHDF5}/lib -L${EBDEVELCUDA}/lib64 -lhdf5 -lhdf5_hl -lz -lsz -lcufft -lcudart

# Build Targets
.build-conf: ${BUILD_SUBPROJECTS}
	"${MAKE}"  -f nbproject/Makefile-${CND_CONF}.mk ${CND_DISTDIR}/${CND_CONF}/k-wave-fluid-cuda

${CND_DISTDIR}/${CND_CONF}/k-wave-fluid-cuda: ${OBJECTFILES}
	${MKDIR} -p ${CND_DISTDIR}/${CND_CONF}
	nvcc -o ${CND_DISTDIR}/${CND_CONF}/k-wave-fluid-cuda ${OBJECTFILES} ${LDLIBSOPTIONS} -Xcompiler="-Wall -O3 -fopenmp -mavx" -arch=compute_52 -code=sm_52

${OBJECTDIR}/Containers/MatrixContainer.o: Containers/MatrixContainer.cpp
	${MKDIR} -p ${OBJECTDIR}/Containers
	$(COMPILE.cc) -O3 -I./ -I${EBROOTHDF5}/include -std=c++11 -o ${OBJECTDIR}/Containers/MatrixContainer.o Containers/MatrixContainer.cpp

${OBJECTDIR}/Containers/MatrixRecord.o: Containers/MatrixRecord.cpp
	${MKDIR} -p ${OBJECTDIR}/Containers
	$(COMPILE.cc) -O3 -I./ -I${EBROOTHDF5}/include -std=c++11 -o ${OBJECTDIR}/Containers/MatrixRecord.o Containers/MatrixRecord.cpp

${OBJECTDIR}/Containers/OutputStreamContainer.o: Containers/OutputStreamContainer.cpp
	${MKDIR} -p ${OBJECTDIR}/Containers
	$(COMPILE.cc) -O3 -I./ -I${EBROOTHDF5}/include -std=c++11 -o ${OBJECTDIR}/Containers/OutputStreamContainer.o Containers/OutputStreamContainer.cpp

${OBJECTDIR}/Hdf5/Hdf5File.o: Hdf5/Hdf5File.cpp
	${MKDIR} -p ${OBJECTDIR}/Hdf5
	$(COMPILE.cc) -O3 -I./ -I${EBROOTHDF5}/include -std=c++11 -o ${OBJECTDIR}/Hdf5/Hdf5File.o Hdf5/Hdf5File.cpp

${OBJECTDIR}/Hdf5/Hdf5FileHeader.o: Hdf5/Hdf5FileHeader.cpp
	${MKDIR} -p ${OBJECTDIR}/Hdf5
	$(COMPILE.cc) -O3 -I./ -I${EBROOTHDF5}/include -std=c++11 -o ${OBJECTDIR}/Hdf5/Hdf5FileHeader.o Hdf5/Hdf5FileHeader.cpp

${OBJECTDIR}/KSpaceSolver/KSpaceFirstOrder3DSolver.o: KSpaceSolver/KSpaceFirstOrder3DSolver.cpp
	${MKDIR} -p ${OBJECTDIR}/KSpaceSolver
	$(COMPILE.cc) -O3 -I./ -I${EBROOTHDF5}/include -std=c++11 -o ${OBJECTDIR}/KSpaceSolver/KSpaceFirstOrder3DSolver.o KSpaceSolver/KSpaceFirstOrder3DSolver.cpp

${OBJECTDIR}/KSpaceSolver/SolverCudaKernels.o: KSpaceSolver/SolverCudaKernels.cu
	${MKDIR} -p ${OBJECTDIR}/KSpaceSolver
	$(COMPILE.cc) -O3 -I./ -I${EBROOTHDF5}/include -std=c++11 -o ${OBJECTDIR}/KSpaceSolver/SolverCudaKernels.o KSpaceSolver/SolverCudaKernels.cu

${OBJECTDIR}/Logger/Logger.o: Logger/Logger.cpp
	${MKDIR} -p ${OBJECTDIR}/Logger
	$(COMPILE.cc) -O3 -I./ -I${EBROOTHDF5}/include -std=c++11 -o ${OBJECTDIR}/Logger/Logger.o Logger/Logger.cpp

${OBJECTDIR}/MatrixClasses/BaseFloatMatrix.o: MatrixClasses/BaseFloatMatrix.cpp
	${MKDIR} -p ${OBJECTDIR}/MatrixClasses
	$(COMPILE.cc) -O3 -I./ -I${EBROOTHDF5}/include -std=c++11 -o ${OBJECTDIR}/MatrixClasses/BaseFloatMatrix.o MatrixClasses/BaseFloatMatrix.cpp

${OBJECTDIR}/MatrixClasses/BaseIndexMatrix.o: MatrixClasses/BaseIndexMatrix.cpp
	${MKDIR} -p ${OBJECTDIR}/MatrixClasses
	$(COMPILE.cc) -O3 -I./ -I${EBROOTHDF5}/include -std=c++11 -o ${OBJECTDIR}/MatrixClasses/BaseIndexMatrix.o MatrixClasses/BaseIndexMatrix.cpp

${OBJECTDIR}/MatrixClasses/ComplexMatrix.o: MatrixClasses/ComplexMatrix.cpp
	${MKDIR} -p ${OBJECTDIR}/MatrixClasses
	$(COMPILE.cc) -O3 -I./ -I${EBROOTHDF5}/include -std=c++11 -o ${OBJECTDIR}/MatrixClasses/ComplexMatrix.o MatrixClasses/ComplexMatrix.cpp

${OBJECTDIR}/MatrixClasses/CufftComplexMatrix.o: MatrixClasses/CufftComplexMatrix.cpp
	${MKDIR} -p ${OBJECTDIR}/MatrixClasses
	$(COMPILE.cc) -O3 -I./ -I${EBROOTHDF5}/include -std=c++11 -o ${OBJECTDIR}/MatrixClasses/CufftComplexMatrix.o MatrixClasses/CufftComplexMatrix.cpp

${OBJECTDIR}/MatrixClasses/IndexMatrix.o: MatrixClasses/IndexMatrix.cpp
	${MKDIR} -p ${OBJECTDIR}/MatrixClasses
	$(COMPILE.cc) -O3 -I./ -I${EBROOTHDF5}/include -std=c++11 -o ${OBJECTDIR}/MatrixClasses/IndexMatrix.o MatrixClasses/IndexMatrix.cpp

${OBJECTDIR}/MatrixClasses/RealMatrix.o: MatrixClasses/RealMatrix.cpp
	${MKDIR} -p ${OBJECTDIR}/MatrixClasses
	$(COMPILE.cc) -O3 -I./ -I${EBROOTHDF5}/include -std=c++11 -o ${OBJECTDIR}/MatrixClasses/RealMatrix.o MatrixClasses/RealMatrix.cpp

${OBJECTDIR}/OutputStreams/BaseOutputStream.o: OutputStreams/BaseOutputStream.cpp
	${MKDIR} -p ${OBJECTDIR}/OutputStreams
	$(COMPILE.cc) -O3 -I./ -I${EBROOTHDF5}/include -std=c++11 -o ${OBJECTDIR}/OutputStreams/BaseOutputStream.o OutputStreams/BaseOutputStream.cpp

${OBJECTDIR}/OutputStreams/CuboidOutputStream.o: OutputStreams/CuboidOutputStream.cpp
	${MKDIR} -p ${OBJECTDIR}/OutputStreams
	$(COMPILE.cc) -O3 -I./ -I${EBROOTHDF5}/include -std=c++11 -o ${OBJECTDIR}/OutputStreams/CuboidOutputStream.o OutputStreams/CuboidOutputStream.cpp

${OBJECTDIR}/OutputStreams/IndexOutputStream.o: OutputStreams/IndexOutputStream.cpp
	${MKDIR} -p ${OBJECTDIR}/OutputStreams
	$(COMPILE.cc) -O3 -I./ -I${EBROOTHDF5}/include -std=c++11 -o ${OBJECTDIR}/OutputStreams/IndexOutputStream.o OutputStreams/IndexOutputStream.cpp

${OBJECTDIR}/OutputStreams/OutputStreamsCudaKernels.o: OutputStreams/OutputStreamsCudaKernels.cu
	${MKDIR} -p ${OBJECTDIR}/OutputStreams
	$(COMPILE.cc) -O3 -I./ -I${EBROOTHDF5}/include -std=c++11 -o ${OBJECTDIR}/OutputStreams/OutputStreamsCudaKernels.o OutputStreams/OutputStreamsCudaKernels.cu

${OBJECTDIR}/OutputStreams/WholeDomainOutputStream.o: OutputStreams/WholeDomainOutputStream.cpp
	${MKDIR} -p ${OBJECTDIR}/OutputStreams
	$(COMPILE.cc) -O3 -I./ -I${EBROOTHDF5}/include -std=c++11 -o ${OBJECTDIR}/OutputStreams/WholeDomainOutputStream.o OutputStreams/WholeDomainOutputStream.cpp

${OBJECTDIR}/Parameters/CommandLineParameters.o: Parameters/CommandLineParameters.cpp
	${MKDIR} -p ${OBJECTDIR}/Parameters
	$(COMPILE.cc) -O3 -I./ -I${EBROOTHDF5}/include -std=c++11 -o ${OBJECTDIR}/Parameters/CommandLineParameters.o Parameters/CommandLineParameters.cpp

${OBJECTDIR}/Parameters/CudaDeviceConstants.o: Parameters/CudaDeviceConstants.cu
	${MKDIR} -p ${OBJECTDIR}/Parameters
	$(COMPILE.cc) -O3 -I./ -I${EBROOTHDF5}/include -std=c++11 -o ${OBJECTDIR}/Parameters/CudaDeviceConstants.o Parameters/CudaDeviceConstants.cu

${OBJECTDIR}/Parameters/CudaParameters.o: Parameters/CudaParameters.cpp
	${MKDIR} -p ${OBJECTDIR}/Parameters
	$(COMPILE.cc) -O3 -I./ -I${EBROOTHDF5}/include -std=c++11 -o ${OBJECTDIR}/Parameters/CudaParameters.o Parameters/CudaParameters.cpp

${OBJECTDIR}/Parameters/Parameters.o: Parameters/Parameters.cpp
	${MKDIR} -p ${OBJECTDIR}/Parameters
	$(COMPILE.cc) -O3 -I./ -I${EBROOTHDF5}/include -std=c++11 -o ${OBJECTDIR}/Parameters/Parameters.o Parameters/Parameters.cpp

${OBJECTDIR}/main.o: main.cpp
	${MKDIR} -p ${OBJECTDIR}
	$(COMPILE.cc) -O3 -I./ -I${EBROOTHDF5}/include -std=c++11 -o ${OBJECTDIR}/main.o main.cpp

# Subprojects
.build-subprojects:

# Clean Targets
.clean-conf: ${CLEAN_SUBPROJECTS}
	${RM} -r ${CND_BUILDDIR}/${CND_CONF}

# Subprojects
.clean-subprojects:
