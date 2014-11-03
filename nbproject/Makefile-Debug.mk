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
CND_CONF=Debug
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
	${OBJECTDIR}/CUDA/MatrixClasses/CUFFTComplexMatrix.o \
	${OBJECTDIR}/HDF5/HDF5_File.o \
	${OBJECTDIR}/KSpaceSolver/KSpaceFirstOrder3DSolver.o \
	${OBJECTDIR}/MatrixClasses/BaseFloatMatrix.o \
	${OBJECTDIR}/MatrixClasses/BaseLongMatrix.o \
	${OBJECTDIR}/MatrixClasses/ComplexMatrix.o \
	${OBJECTDIR}/MatrixClasses/LongMatrix.o \
	${OBJECTDIR}/MatrixClasses/MatrixContainer.o \
	${OBJECTDIR}/MatrixClasses/MatrixRecord.o \
	${OBJECTDIR}/MatrixClasses/OutputHDF5Stream/BaseOutputHDF5Stream.o \
	${OBJECTDIR}/MatrixClasses/OutputHDF5Stream/IndexOutputHDF5Stream.o \
	${OBJECTDIR}/MatrixClasses/OutputHDF5Stream/WholeDomainOutputHDF5Stream.o \
	${OBJECTDIR}/MatrixClasses/OutputStreamContainer.o \
	${OBJECTDIR}/MatrixClasses/RealMatrix.o \
	${OBJECTDIR}/Parameters/CommandLineParameters.o \
	${OBJECTDIR}/Parameters/Parameters.o \
	${OBJECTDIR}/main.o


# C Compiler Flags
CFLAGS=

# CC Compiler Flags
CCFLAGS=-Xcompiler="-fopenmp"
CXXFLAGS=-Xcompiler="-fopenmp"

# Fortran Compiler Flags
FFLAGS=

# Assembler Flags
ASFLAGS=

# Link Libraries and Options
LDLIBSOPTIONS=-L/usr/local/hdf5-serial/lib -L/usr/local/cuda/lib64 /usr/local/hdf5-serial/lib/libhdf5_hl.a /usr/local/hdf5-serial/lib/libhdf5.a -lz -lcufft

# Build Targets
.build-conf: ${BUILD_SUBPROJECTS}
	"${MAKE}"  -f nbproject/Makefile-${CND_CONF}.mk ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/k-wave-fluid-cuda

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/k-wave-fluid-cuda: /usr/local/hdf5-serial/lib/libhdf5_hl.a

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/k-wave-fluid-cuda: /usr/local/hdf5-serial/lib/libhdf5.a

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/k-wave-fluid-cuda: ${OBJECTFILES}
	${MKDIR} -p ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}
	${LINK.cc} -o ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/k-wave-fluid-cuda ${OBJECTFILES} ${LDLIBSOPTIONS}

${OBJECTDIR}/CUDA/CUDAImplementations.o: CUDA/CUDAImplementations.cu 
	${MKDIR} -p ${OBJECTDIR}/CUDA
	$(COMPILE.cc) -g -w -DCUDA_VERSION=1 -DNDEBUG -I/usr/local/hdf5-serial/include -o ${OBJECTDIR}/CUDA/CUDAImplementations.o CUDA/CUDAImplementations.cu

${OBJECTDIR}/CUDA/CUDATuner.o: CUDA/CUDATuner.cpp 
	${MKDIR} -p ${OBJECTDIR}/CUDA
	$(COMPILE.cc) -g -w -DCUDA_VERSION=1 -DNDEBUG -I/usr/local/hdf5-serial/include -o ${OBJECTDIR}/CUDA/CUDATuner.o CUDA/CUDATuner.cpp

${OBJECTDIR}/CUDA/MatrixClasses/CUFFTComplexMatrix.o: CUDA/MatrixClasses/CUFFTComplexMatrix.cpp 
	${MKDIR} -p ${OBJECTDIR}/CUDA/MatrixClasses
	$(COMPILE.cc) -g -w -DCUDA_VERSION=1 -DNDEBUG -I/usr/local/hdf5-serial/include -o ${OBJECTDIR}/CUDA/MatrixClasses/CUFFTComplexMatrix.o CUDA/MatrixClasses/CUFFTComplexMatrix.cpp

${OBJECTDIR}/HDF5/HDF5_File.o: HDF5/HDF5_File.cpp 
	${MKDIR} -p ${OBJECTDIR}/HDF5
	$(COMPILE.cc) -g -w -DCUDA_VERSION=1 -DNDEBUG -I/usr/local/hdf5-serial/include -o ${OBJECTDIR}/HDF5/HDF5_File.o HDF5/HDF5_File.cpp

${OBJECTDIR}/KSpaceSolver/KSpaceFirstOrder3DSolver.o: KSpaceSolver/KSpaceFirstOrder3DSolver.cpp 
	${MKDIR} -p ${OBJECTDIR}/KSpaceSolver
	$(COMPILE.cc) -g -w -DCUDA_VERSION=1 -DNDEBUG -I/usr/local/hdf5-serial/include -o ${OBJECTDIR}/KSpaceSolver/KSpaceFirstOrder3DSolver.o KSpaceSolver/KSpaceFirstOrder3DSolver.cpp

${OBJECTDIR}/MatrixClasses/BaseFloatMatrix.o: MatrixClasses/BaseFloatMatrix.cpp 
	${MKDIR} -p ${OBJECTDIR}/MatrixClasses
	$(COMPILE.cc) -g -w -DCUDA_VERSION=1 -DNDEBUG -I/usr/local/hdf5-serial/include -o ${OBJECTDIR}/MatrixClasses/BaseFloatMatrix.o MatrixClasses/BaseFloatMatrix.cpp

${OBJECTDIR}/MatrixClasses/BaseLongMatrix.o: MatrixClasses/BaseLongMatrix.cpp 
	${MKDIR} -p ${OBJECTDIR}/MatrixClasses
	$(COMPILE.cc) -g -w -DCUDA_VERSION=1 -DNDEBUG -I/usr/local/hdf5-serial/include -o ${OBJECTDIR}/MatrixClasses/BaseLongMatrix.o MatrixClasses/BaseLongMatrix.cpp

${OBJECTDIR}/MatrixClasses/ComplexMatrix.o: MatrixClasses/ComplexMatrix.cpp 
	${MKDIR} -p ${OBJECTDIR}/MatrixClasses
	$(COMPILE.cc) -g -w -DCUDA_VERSION=1 -DNDEBUG -I/usr/local/hdf5-serial/include -o ${OBJECTDIR}/MatrixClasses/ComplexMatrix.o MatrixClasses/ComplexMatrix.cpp

${OBJECTDIR}/MatrixClasses/LongMatrix.o: MatrixClasses/LongMatrix.cpp 
	${MKDIR} -p ${OBJECTDIR}/MatrixClasses
	$(COMPILE.cc) -g -w -DCUDA_VERSION=1 -DNDEBUG -I/usr/local/hdf5-serial/include -o ${OBJECTDIR}/MatrixClasses/LongMatrix.o MatrixClasses/LongMatrix.cpp

${OBJECTDIR}/MatrixClasses/MatrixContainer.o: MatrixClasses/MatrixContainer.cpp 
	${MKDIR} -p ${OBJECTDIR}/MatrixClasses
	$(COMPILE.cc) -g -w -DCUDA_VERSION=1 -DNDEBUG -I/usr/local/hdf5-serial/include -o ${OBJECTDIR}/MatrixClasses/MatrixContainer.o MatrixClasses/MatrixContainer.cpp

${OBJECTDIR}/MatrixClasses/MatrixRecord.o: MatrixClasses/MatrixRecord.cpp 
	${MKDIR} -p ${OBJECTDIR}/MatrixClasses
	$(COMPILE.cc) -g -w -DCUDA_VERSION=1 -DNDEBUG -I/usr/local/hdf5-serial/include -o ${OBJECTDIR}/MatrixClasses/MatrixRecord.o MatrixClasses/MatrixRecord.cpp

${OBJECTDIR}/MatrixClasses/OutputHDF5Stream/BaseOutputHDF5Stream.o: MatrixClasses/OutputHDF5Stream/BaseOutputHDF5Stream.cpp 
	${MKDIR} -p ${OBJECTDIR}/MatrixClasses/OutputHDF5Stream
	$(COMPILE.cc) -g -w -DCUDA_VERSION=1 -DNDEBUG -I/usr/local/hdf5-serial/include -o ${OBJECTDIR}/MatrixClasses/OutputHDF5Stream/BaseOutputHDF5Stream.o MatrixClasses/OutputHDF5Stream/BaseOutputHDF5Stream.cpp

${OBJECTDIR}/MatrixClasses/OutputHDF5Stream/IndexOutputHDF5Stream.o: MatrixClasses/OutputHDF5Stream/IndexOutputHDF5Stream.cpp 
	${MKDIR} -p ${OBJECTDIR}/MatrixClasses/OutputHDF5Stream
	$(COMPILE.cc) -g -w -DCUDA_VERSION=1 -DNDEBUG -I/usr/local/hdf5-serial/include -o ${OBJECTDIR}/MatrixClasses/OutputHDF5Stream/IndexOutputHDF5Stream.o MatrixClasses/OutputHDF5Stream/IndexOutputHDF5Stream.cpp

${OBJECTDIR}/MatrixClasses/OutputHDF5Stream/WholeDomainOutputHDF5Stream.o: MatrixClasses/OutputHDF5Stream/WholeDomainOutputHDF5Stream.cpp 
	${MKDIR} -p ${OBJECTDIR}/MatrixClasses/OutputHDF5Stream
	$(COMPILE.cc) -g -w -DCUDA_VERSION=1 -DNDEBUG -I/usr/local/hdf5-serial/include -o ${OBJECTDIR}/MatrixClasses/OutputHDF5Stream/WholeDomainOutputHDF5Stream.o MatrixClasses/OutputHDF5Stream/WholeDomainOutputHDF5Stream.cpp

${OBJECTDIR}/MatrixClasses/OutputStreamContainer.o: MatrixClasses/OutputStreamContainer.cpp 
	${MKDIR} -p ${OBJECTDIR}/MatrixClasses
	$(COMPILE.cc) -g -w -DCUDA_VERSION=1 -DNDEBUG -I/usr/local/hdf5-serial/include -o ${OBJECTDIR}/MatrixClasses/OutputStreamContainer.o MatrixClasses/OutputStreamContainer.cpp

${OBJECTDIR}/MatrixClasses/RealMatrix.o: MatrixClasses/RealMatrix.cpp 
	${MKDIR} -p ${OBJECTDIR}/MatrixClasses
	$(COMPILE.cc) -g -w -DCUDA_VERSION=1 -DNDEBUG -I/usr/local/hdf5-serial/include -o ${OBJECTDIR}/MatrixClasses/RealMatrix.o MatrixClasses/RealMatrix.cpp

${OBJECTDIR}/Parameters/CommandLineParameters.o: Parameters/CommandLineParameters.cpp 
	${MKDIR} -p ${OBJECTDIR}/Parameters
	$(COMPILE.cc) -g -w -DCUDA_VERSION=1 -DNDEBUG -I/usr/local/hdf5-serial/include -o ${OBJECTDIR}/Parameters/CommandLineParameters.o Parameters/CommandLineParameters.cpp

${OBJECTDIR}/Parameters/Parameters.o: Parameters/Parameters.cpp 
	${MKDIR} -p ${OBJECTDIR}/Parameters
	$(COMPILE.cc) -g -w -DCUDA_VERSION=1 -DNDEBUG -I/usr/local/hdf5-serial/include -o ${OBJECTDIR}/Parameters/Parameters.o Parameters/Parameters.cpp

${OBJECTDIR}/main.o: main.cpp 
	${MKDIR} -p ${OBJECTDIR}
	$(COMPILE.cc) -g -w -DCUDA_VERSION=1 -DNDEBUG -I/usr/local/hdf5-serial/include -o ${OBJECTDIR}/main.o main.cpp

# Subprojects
.build-subprojects:

# Clean Targets
.clean-conf: ${CLEAN_SUBPROJECTS}
	${RM} -r ${CND_BUILDDIR}/${CND_CONF}
	${RM} ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/k-wave-fluid-cuda

# Subprojects
.clean-subprojects:
