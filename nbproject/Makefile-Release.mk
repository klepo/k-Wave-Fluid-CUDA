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
CND_PLATFORM=CUDA-Linux-x86
CND_DLIB_EXT=so
CND_CONF=Release
CND_DISTDIR=dist
CND_BUILDDIR=build

# Include project Makefile
include Makefile

# Object Directory
OBJECTDIR=${CND_BUILDDIR}/${CND_CONF}/${CND_PLATFORM}

# Object Files
OBJECTFILES= \
	${OBJECTDIR}/CUDA/CUDAImplementations.o \
	${OBJECTDIR}/CUDA/CUDATuner.o \
	${OBJECTDIR}/Containers/MatrixContainer.o \
	${OBJECTDIR}/Containers/MatrixRecord.o \
	${OBJECTDIR}/Containers/OutputStreamContainer.o \
	${OBJECTDIR}/HDF5/HDF5_File.o \
	${OBJECTDIR}/KSpaceSolver/KSpaceFirstOrder3DSolver.o \
	${OBJECTDIR}/MatrixClasses/BaseFloatMatrix.o \
	${OBJECTDIR}/MatrixClasses/BaseIndexMatrix.o \
	${OBJECTDIR}/MatrixClasses/CUFFTComplexMatrix.o \
	${OBJECTDIR}/MatrixClasses/ComplexMatrix.o \
	${OBJECTDIR}/MatrixClasses/IndexMatrix.o \
	${OBJECTDIR}/MatrixClasses/RealMatrix.o \
	${OBJECTDIR}/OutputHDF5Streams/BaseOutputHDF5Stream.o \
	${OBJECTDIR}/OutputHDF5Streams/IndexOutputHDF5Stream.o \
	${OBJECTDIR}/OutputHDF5Streams/OutputStreamsCUDAKernels.o \
	${OBJECTDIR}/OutputHDF5Streams/WholeDomainOutputHDF5Stream.o \
	${OBJECTDIR}/Parameters/CommandLineParameters.o \
	${OBJECTDIR}/Parameters/Parameters.o \
	${OBJECTDIR}/main.o


# C Compiler Flags
CFLAGS=

# CC Compiler Flags
CCFLAGS=
CXXFLAGS=

# Fortran Compiler Flags
FFLAGS=

# Assembler Flags
ASFLAGS=

# Link Libraries and Options
LDLIBSOPTIONS=

# Build Targets
.build-conf: ${BUILD_SUBPROJECTS}
	"${MAKE}"  -f nbproject/Makefile-${CND_CONF}.mk ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/k-wave-fluid-cuda

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/k-wave-fluid-cuda: ${OBJECTFILES}
	${MKDIR} -p ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}
	${LINK.cc} -o ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/k-wave-fluid-cuda ${OBJECTFILES} ${LDLIBSOPTIONS}

${OBJECTDIR}/CUDA/CUDAImplementations.o: CUDA/CUDAImplementations.cu 
	${MKDIR} -p ${OBJECTDIR}/CUDA
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/CUDA/CUDAImplementations.o CUDA/CUDAImplementations.cu

${OBJECTDIR}/CUDA/CUDATuner.o: CUDA/CUDATuner.cpp 
	${MKDIR} -p ${OBJECTDIR}/CUDA
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/CUDA/CUDATuner.o CUDA/CUDATuner.cpp

${OBJECTDIR}/Containers/MatrixContainer.o: Containers/MatrixContainer.cpp 
	${MKDIR} -p ${OBJECTDIR}/Containers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/Containers/MatrixContainer.o Containers/MatrixContainer.cpp

${OBJECTDIR}/Containers/MatrixRecord.o: Containers/MatrixRecord.cpp 
	${MKDIR} -p ${OBJECTDIR}/Containers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/Containers/MatrixRecord.o Containers/MatrixRecord.cpp

${OBJECTDIR}/Containers/OutputStreamContainer.o: Containers/OutputStreamContainer.cpp 
	${MKDIR} -p ${OBJECTDIR}/Containers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/Containers/OutputStreamContainer.o Containers/OutputStreamContainer.cpp

${OBJECTDIR}/HDF5/HDF5_File.o: HDF5/HDF5_File.cpp 
	${MKDIR} -p ${OBJECTDIR}/HDF5
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/HDF5/HDF5_File.o HDF5/HDF5_File.cpp

${OBJECTDIR}/KSpaceSolver/KSpaceFirstOrder3DSolver.o: KSpaceSolver/KSpaceFirstOrder3DSolver.cpp 
	${MKDIR} -p ${OBJECTDIR}/KSpaceSolver
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/KSpaceSolver/KSpaceFirstOrder3DSolver.o KSpaceSolver/KSpaceFirstOrder3DSolver.cpp

${OBJECTDIR}/MatrixClasses/BaseFloatMatrix.o: MatrixClasses/BaseFloatMatrix.cpp 
	${MKDIR} -p ${OBJECTDIR}/MatrixClasses
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/MatrixClasses/BaseFloatMatrix.o MatrixClasses/BaseFloatMatrix.cpp

${OBJECTDIR}/MatrixClasses/BaseIndexMatrix.o: MatrixClasses/BaseIndexMatrix.cpp 
	${MKDIR} -p ${OBJECTDIR}/MatrixClasses
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/MatrixClasses/BaseIndexMatrix.o MatrixClasses/BaseIndexMatrix.cpp

${OBJECTDIR}/MatrixClasses/CUFFTComplexMatrix.o: MatrixClasses/CUFFTComplexMatrix.cpp 
	${MKDIR} -p ${OBJECTDIR}/MatrixClasses
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/MatrixClasses/CUFFTComplexMatrix.o MatrixClasses/CUFFTComplexMatrix.cpp

${OBJECTDIR}/MatrixClasses/ComplexMatrix.o: MatrixClasses/ComplexMatrix.cpp 
	${MKDIR} -p ${OBJECTDIR}/MatrixClasses
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/MatrixClasses/ComplexMatrix.o MatrixClasses/ComplexMatrix.cpp

${OBJECTDIR}/MatrixClasses/IndexMatrix.o: MatrixClasses/IndexMatrix.cpp 
	${MKDIR} -p ${OBJECTDIR}/MatrixClasses
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/MatrixClasses/IndexMatrix.o MatrixClasses/IndexMatrix.cpp

${OBJECTDIR}/MatrixClasses/RealMatrix.o: MatrixClasses/RealMatrix.cpp 
	${MKDIR} -p ${OBJECTDIR}/MatrixClasses
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/MatrixClasses/RealMatrix.o MatrixClasses/RealMatrix.cpp

${OBJECTDIR}/OutputHDF5Streams/BaseOutputHDF5Stream.o: OutputHDF5Streams/BaseOutputHDF5Stream.cpp 
	${MKDIR} -p ${OBJECTDIR}/OutputHDF5Streams
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/OutputHDF5Streams/BaseOutputHDF5Stream.o OutputHDF5Streams/BaseOutputHDF5Stream.cpp

${OBJECTDIR}/OutputHDF5Streams/IndexOutputHDF5Stream.o: OutputHDF5Streams/IndexOutputHDF5Stream.cpp 
	${MKDIR} -p ${OBJECTDIR}/OutputHDF5Streams
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/OutputHDF5Streams/IndexOutputHDF5Stream.o OutputHDF5Streams/IndexOutputHDF5Stream.cpp

${OBJECTDIR}/OutputHDF5Streams/OutputStreamsCUDAKernels.o: OutputHDF5Streams/OutputStreamsCUDAKernels.cu 
	${MKDIR} -p ${OBJECTDIR}/OutputHDF5Streams
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/OutputHDF5Streams/OutputStreamsCUDAKernels.o OutputHDF5Streams/OutputStreamsCUDAKernels.cu

${OBJECTDIR}/OutputHDF5Streams/WholeDomainOutputHDF5Stream.o: OutputHDF5Streams/WholeDomainOutputHDF5Stream.cpp 
	${MKDIR} -p ${OBJECTDIR}/OutputHDF5Streams
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/OutputHDF5Streams/WholeDomainOutputHDF5Stream.o OutputHDF5Streams/WholeDomainOutputHDF5Stream.cpp

${OBJECTDIR}/Parameters/CommandLineParameters.o: Parameters/CommandLineParameters.cpp 
	${MKDIR} -p ${OBJECTDIR}/Parameters
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/Parameters/CommandLineParameters.o Parameters/CommandLineParameters.cpp

${OBJECTDIR}/Parameters/Parameters.o: Parameters/Parameters.cpp 
	${MKDIR} -p ${OBJECTDIR}/Parameters
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/Parameters/Parameters.o Parameters/Parameters.cpp

${OBJECTDIR}/main.o: main.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/main.o main.cpp

# Subprojects
.build-subprojects:

# Clean Targets
.clean-conf: ${CLEAN_SUBPROJECTS}
	${RM} -r ${CND_BUILDDIR}/${CND_CONF}
	${RM} ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/k-wave-fluid-cuda

# Subprojects
.clean-subprojects:

# Enable dependency checking
.dep.inc: .depcheck-impl

include .dep.inc
