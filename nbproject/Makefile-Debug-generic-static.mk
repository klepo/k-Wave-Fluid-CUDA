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
CND_PLATFORM=GNU-Linux
CND_DLIB_EXT=so
CND_CONF=Debug-generic-static
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
	${OBJECTDIR}/HDF5/HDF5_File.o \
	${OBJECTDIR}/KSpaceSolver/KSpaceFirstOrder3DSolver.o \
	${OBJECTDIR}/KSpaceSolver/SolverCUDAKernels.o \
	${OBJECTDIR}/Logger/Logger.o \
	${OBJECTDIR}/MatrixClasses/BaseFloatMatrix.o \
	${OBJECTDIR}/MatrixClasses/BaseIndexMatrix.o \
	${OBJECTDIR}/MatrixClasses/CUFFTComplexMatrix.o \
	${OBJECTDIR}/MatrixClasses/ComplexMatrix.o \
	${OBJECTDIR}/MatrixClasses/IndexMatrix.o \
	${OBJECTDIR}/MatrixClasses/RealMatrix.o \
	${OBJECTDIR}/OutputHDF5Streams/BaseOutputHDF5Stream.o \
	${OBJECTDIR}/OutputHDF5Streams/CuboidOutputHDF5Stream.o \
	${OBJECTDIR}/OutputHDF5Streams/IndexOutputHDF5Stream.o \
	${OBJECTDIR}/OutputHDF5Streams/OutputStreamsCUDAKernels.o \
	${OBJECTDIR}/OutputHDF5Streams/WholeDomainOutputHDF5Stream.o \
	${OBJECTDIR}/Parameters/CUDADeviceConstants.o \
	${OBJECTDIR}/Parameters/CUDAParameters.o \
	${OBJECTDIR}/Parameters/CommandLineParameters.o \
	${OBJECTDIR}/Parameters/Parameters.o \
	${OBJECTDIR}/main.o


# C Compiler Flags
CFLAGS=-m64

# CC Compiler Flags
CCFLAGS=-m64 -Xcompiler=" -g3 -gdwarf-2  -Wall -fopenmp" -G --device-c
CXXFLAGS=-m64 -Xcompiler=" -g3 -gdwarf-2  -Wall -fopenmp" -G --device-c

# Fortran Compiler Flags
FFLAGS=

# Assembler Flags
ASFLAGS=

# Link Libraries and Options
LDLIBSOPTIONS=-L${EBROOTHDF5}/lib -L${EBDEVELCUDA}/lib64 ${EBROOTHDF5}/lib/libhdf5_hl.a ${EBROOTHDF5}/lib/libhdf5.a ${EBROOTSZIP}/lib/libsz.a ${EBROOTZLIB}/lib/libz.a ${EBROOTCUDA}/lib64/libcufft_static.a ${EBROOTCUDA}/lib64/libculibos.a ${EBROOTCUDA}/lib64/libcudart_static.a

# Build Targets
.build-conf: ${BUILD_SUBPROJECTS}
	"${MAKE}"  -f nbproject/Makefile-${CND_CONF}.mk ${CND_DISTDIR}/${CND_CONF}/k-wave-fluid-cuda

${CND_DISTDIR}/${CND_CONF}/k-wave-fluid-cuda: ${EBROOTHDF5}/lib/libhdf5_hl.a

${CND_DISTDIR}/${CND_CONF}/k-wave-fluid-cuda: ${EBROOTHDF5}/lib/libhdf5.a

${CND_DISTDIR}/${CND_CONF}/k-wave-fluid-cuda: ${EBROOTSZIP}/lib/libsz.a

${CND_DISTDIR}/${CND_CONF}/k-wave-fluid-cuda: ${EBROOTZLIB}/lib/libz.a

${CND_DISTDIR}/${CND_CONF}/k-wave-fluid-cuda: ${EBROOTCUDA}/lib64/libcufft_static.a

${CND_DISTDIR}/${CND_CONF}/k-wave-fluid-cuda: ${EBROOTCUDA}/lib64/libculibos.a

${CND_DISTDIR}/${CND_CONF}/k-wave-fluid-cuda: ${EBROOTCUDA}/lib64/libcudart_static.a

${CND_DISTDIR}/${CND_CONF}/k-wave-fluid-cuda: ${OBJECTFILES}
	${MKDIR} -p ${CND_DISTDIR}/${CND_CONF}
	nvcc -o ${CND_DISTDIR}/${CND_CONF}/k-wave-fluid-cuda ${OBJECTFILES} ${LDLIBSOPTIONS} -Xcompiler="-g3 -gdwarf-2 -Wall -fopenmp" -G

${OBJECTDIR}/Containers/MatrixContainer.o: Containers/MatrixContainer.cpp
	${MKDIR} -p ${OBJECTDIR}/Containers
	$(COMPILE.cc) -g -I./ -I${EBROOTHDF5}/include -std=c++11 -o ${OBJECTDIR}/Containers/MatrixContainer.o Containers/MatrixContainer.cpp

${OBJECTDIR}/Containers/MatrixRecord.o: Containers/MatrixRecord.cpp
	${MKDIR} -p ${OBJECTDIR}/Containers
	$(COMPILE.cc) -g -I./ -I${EBROOTHDF5}/include -std=c++11 -o ${OBJECTDIR}/Containers/MatrixRecord.o Containers/MatrixRecord.cpp

${OBJECTDIR}/Containers/OutputStreamContainer.o: Containers/OutputStreamContainer.cpp
	${MKDIR} -p ${OBJECTDIR}/Containers
	$(COMPILE.cc) -g -I./ -I${EBROOTHDF5}/include -std=c++11 -o ${OBJECTDIR}/Containers/OutputStreamContainer.o Containers/OutputStreamContainer.cpp

${OBJECTDIR}/HDF5/HDF5_File.o: HDF5/HDF5_File.cpp
	${MKDIR} -p ${OBJECTDIR}/HDF5
	$(COMPILE.cc) -g -I./ -I${EBROOTHDF5}/include -std=c++11 -o ${OBJECTDIR}/HDF5/HDF5_File.o HDF5/HDF5_File.cpp

${OBJECTDIR}/KSpaceSolver/KSpaceFirstOrder3DSolver.o: KSpaceSolver/KSpaceFirstOrder3DSolver.cpp
	${MKDIR} -p ${OBJECTDIR}/KSpaceSolver
	$(COMPILE.cc) -g -I./ -I${EBROOTHDF5}/include -std=c++11 -o ${OBJECTDIR}/KSpaceSolver/KSpaceFirstOrder3DSolver.o KSpaceSolver/KSpaceFirstOrder3DSolver.cpp

${OBJECTDIR}/KSpaceSolver/SolverCUDAKernels.o: KSpaceSolver/SolverCUDAKernels.cu
	${MKDIR} -p ${OBJECTDIR}/KSpaceSolver
	$(COMPILE.cc) -g -I./ -I${EBROOTHDF5}/include -std=c++11 -o ${OBJECTDIR}/KSpaceSolver/SolverCUDAKernels.o KSpaceSolver/SolverCUDAKernels.cu

${OBJECTDIR}/Logger/Logger.o: Logger/Logger.cpp
	${MKDIR} -p ${OBJECTDIR}/Logger
	$(COMPILE.cc) -g -I./ -I${EBROOTHDF5}/include -std=c++11 -o ${OBJECTDIR}/Logger/Logger.o Logger/Logger.cpp

${OBJECTDIR}/MatrixClasses/BaseFloatMatrix.o: MatrixClasses/BaseFloatMatrix.cpp
	${MKDIR} -p ${OBJECTDIR}/MatrixClasses
	$(COMPILE.cc) -g -I./ -I${EBROOTHDF5}/include -std=c++11 -o ${OBJECTDIR}/MatrixClasses/BaseFloatMatrix.o MatrixClasses/BaseFloatMatrix.cpp

${OBJECTDIR}/MatrixClasses/BaseIndexMatrix.o: MatrixClasses/BaseIndexMatrix.cpp
	${MKDIR} -p ${OBJECTDIR}/MatrixClasses
	$(COMPILE.cc) -g -I./ -I${EBROOTHDF5}/include -std=c++11 -o ${OBJECTDIR}/MatrixClasses/BaseIndexMatrix.o MatrixClasses/BaseIndexMatrix.cpp

${OBJECTDIR}/MatrixClasses/CUFFTComplexMatrix.o: MatrixClasses/CUFFTComplexMatrix.cpp
	${MKDIR} -p ${OBJECTDIR}/MatrixClasses
	$(COMPILE.cc) -g -I./ -I${EBROOTHDF5}/include -std=c++11 -o ${OBJECTDIR}/MatrixClasses/CUFFTComplexMatrix.o MatrixClasses/CUFFTComplexMatrix.cpp

${OBJECTDIR}/MatrixClasses/ComplexMatrix.o: MatrixClasses/ComplexMatrix.cpp
	${MKDIR} -p ${OBJECTDIR}/MatrixClasses
	$(COMPILE.cc) -g -I./ -I${EBROOTHDF5}/include -std=c++11 -o ${OBJECTDIR}/MatrixClasses/ComplexMatrix.o MatrixClasses/ComplexMatrix.cpp

${OBJECTDIR}/MatrixClasses/IndexMatrix.o: MatrixClasses/IndexMatrix.cpp
	${MKDIR} -p ${OBJECTDIR}/MatrixClasses
	$(COMPILE.cc) -g -I./ -I${EBROOTHDF5}/include -std=c++11 -o ${OBJECTDIR}/MatrixClasses/IndexMatrix.o MatrixClasses/IndexMatrix.cpp

${OBJECTDIR}/MatrixClasses/RealMatrix.o: MatrixClasses/RealMatrix.cpp
	${MKDIR} -p ${OBJECTDIR}/MatrixClasses
	$(COMPILE.cc) -g -I./ -I${EBROOTHDF5}/include -std=c++11 -o ${OBJECTDIR}/MatrixClasses/RealMatrix.o MatrixClasses/RealMatrix.cpp

${OBJECTDIR}/OutputHDF5Streams/BaseOutputHDF5Stream.o: OutputHDF5Streams/BaseOutputHDF5Stream.cpp
	${MKDIR} -p ${OBJECTDIR}/OutputHDF5Streams
	$(COMPILE.cc) -g -I./ -I${EBROOTHDF5}/include -std=c++11 -o ${OBJECTDIR}/OutputHDF5Streams/BaseOutputHDF5Stream.o OutputHDF5Streams/BaseOutputHDF5Stream.cpp

${OBJECTDIR}/OutputHDF5Streams/CuboidOutputHDF5Stream.o: OutputHDF5Streams/CuboidOutputHDF5Stream.cpp
	${MKDIR} -p ${OBJECTDIR}/OutputHDF5Streams
	$(COMPILE.cc) -g -I./ -I${EBROOTHDF5}/include -std=c++11 -o ${OBJECTDIR}/OutputHDF5Streams/CuboidOutputHDF5Stream.o OutputHDF5Streams/CuboidOutputHDF5Stream.cpp

${OBJECTDIR}/OutputHDF5Streams/IndexOutputHDF5Stream.o: OutputHDF5Streams/IndexOutputHDF5Stream.cpp
	${MKDIR} -p ${OBJECTDIR}/OutputHDF5Streams
	$(COMPILE.cc) -g -I./ -I${EBROOTHDF5}/include -std=c++11 -o ${OBJECTDIR}/OutputHDF5Streams/IndexOutputHDF5Stream.o OutputHDF5Streams/IndexOutputHDF5Stream.cpp

${OBJECTDIR}/OutputHDF5Streams/OutputStreamsCUDAKernels.o: OutputHDF5Streams/OutputStreamsCUDAKernels.cu
	${MKDIR} -p ${OBJECTDIR}/OutputHDF5Streams
	$(COMPILE.cc) -g -I./ -I${EBROOTHDF5}/include -std=c++11 -o ${OBJECTDIR}/OutputHDF5Streams/OutputStreamsCUDAKernels.o OutputHDF5Streams/OutputStreamsCUDAKernels.cu

${OBJECTDIR}/OutputHDF5Streams/WholeDomainOutputHDF5Stream.o: OutputHDF5Streams/WholeDomainOutputHDF5Stream.cpp
	${MKDIR} -p ${OBJECTDIR}/OutputHDF5Streams
	$(COMPILE.cc) -g -I./ -I${EBROOTHDF5}/include -std=c++11 -o ${OBJECTDIR}/OutputHDF5Streams/WholeDomainOutputHDF5Stream.o OutputHDF5Streams/WholeDomainOutputHDF5Stream.cpp

${OBJECTDIR}/Parameters/CUDADeviceConstants.o: Parameters/CUDADeviceConstants.cu
	${MKDIR} -p ${OBJECTDIR}/Parameters
	$(COMPILE.cc) -g -I./ -I${EBROOTHDF5}/include -std=c++11 -o ${OBJECTDIR}/Parameters/CUDADeviceConstants.o Parameters/CUDADeviceConstants.cu

${OBJECTDIR}/Parameters/CUDAParameters.o: Parameters/CUDAParameters.cpp
	${MKDIR} -p ${OBJECTDIR}/Parameters
	$(COMPILE.cc) -g -I./ -I${EBROOTHDF5}/include -std=c++11 -o ${OBJECTDIR}/Parameters/CUDAParameters.o Parameters/CUDAParameters.cpp

${OBJECTDIR}/Parameters/CommandLineParameters.o: Parameters/CommandLineParameters.cpp
	${MKDIR} -p ${OBJECTDIR}/Parameters
	$(COMPILE.cc) -g -I./ -I${EBROOTHDF5}/include -std=c++11 -o ${OBJECTDIR}/Parameters/CommandLineParameters.o Parameters/CommandLineParameters.cpp

${OBJECTDIR}/Parameters/Parameters.o: Parameters/Parameters.cpp
	${MKDIR} -p ${OBJECTDIR}/Parameters
	$(COMPILE.cc) -g -I./ -I${EBROOTHDF5}/include -std=c++11 -o ${OBJECTDIR}/Parameters/Parameters.o Parameters/Parameters.cpp

${OBJECTDIR}/main.o: main.cpp
	${MKDIR} -p ${OBJECTDIR}
	$(COMPILE.cc) -g -I./ -I${EBROOTHDF5}/include -std=c++11 -o ${OBJECTDIR}/main.o main.cpp

# Subprojects
.build-subprojects:

# Clean Targets
.clean-conf: ${CLEAN_SUBPROJECTS}
	${RM} -r ${CND_BUILDDIR}/${CND_CONF}

# Subprojects
.clean-subprojects:
