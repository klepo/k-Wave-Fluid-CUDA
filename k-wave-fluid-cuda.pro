#-------------------------------------------------
#
# Project created by QtCreator
#
# k-wave-fluid-cuda application
#
#-------------------------------------------------

TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt
CONFIG -= debug_and_release
CONFIG += c++11

TARGET = k-wave-fluid-cuda

win32:QMAKE_LFLAGS += /ignore:4099

INCLUDEPATH += $$PWD
DEPENDPATH += $$PWD
#INCLUDEPATH += $$(MKLROOT)\include\fftw
#DEPENDPATH += $$(MKLROOT)\include\fftw

#LIBS += -L$$(MKLROOT)\lib\intel64

win32 {
#    QMAKE_CXXFLAGS += /Qmkl=parallel
#    QMAKE_CFLAGS += /Qmkl=parallel
    DEFINES -= _UNICODE
}

unix {
#    QMAKE_CXXFLAGS += -mkl=parallel
#    QMAKE_CFLAGS += -mkl=parallel
}

# Check Qt version
lessThan(QT_MAJOR_VERSION, 5) : lessThan(QT_MINOR_VERSION, 2) {
    error(Qt version is too old)
}

# HDF5 library
include($$PWD/qtproject/hdf5.pri)

# OpenMP library
include($$PWD/qtproject/openmp.pri)

SOURCES += \
    main.cpp \
    Compression/CompressHelper.cpp \
    Containers/MatrixContainer.cpp \
    Containers/MatrixRecord.cpp \
    Containers/OutputStreamContainer.cpp \
    GetoptWin64/Getopt.cpp \
    Hdf5/Hdf5File.cpp \
    Hdf5/Hdf5FileHeader.cpp \
    KSpaceSolver/KSpaceFirstOrderSolver.cpp \
    Logger/Logger.cpp \
    MatrixClasses/BaseFloatMatrix.cpp \
    MatrixClasses/BaseIndexMatrix.cpp \
    MatrixClasses/ComplexMatrix.cpp \
    MatrixClasses/CufftComplexMatrix.cpp \
    MatrixClasses/IndexMatrix.cpp \
    MatrixClasses/RealMatrix.cpp \
    OutputStreams/BaseOutputStream.cpp \
    OutputStreams/CuboidOutputStream.cpp \
    OutputStreams/IndexOutputStream.cpp \
    OutputStreams/WholeDomainOutputStream.cpp \
    Parameters/CommandLineParameters.cpp \
    Parameters/Parameters.cpp \
    Parameters/CudaParameters.cpp \

HEADERS += \
    Compression/CompressHelper.h \
    Containers/MatrixContainer.h \
    Containers/MatrixRecord.h \
    Containers/OutputStreamContainer.h \
    GetoptWin64/Getopt.h \
    Hdf5/Hdf5File.h \
    Hdf5/Hdf5FileHeader.h \
    KSpaceSolver/KSpaceFirstOrderSolver.h \
    Logger/ErrorMessages.h \
    Logger/ErrorMessagesLinux.h \
    Logger/ErrorMessagesWindows.h \
    Logger/Logger.h \
    Logger/OutputMessages.h \
    Logger/OutputMessagesLinux.h \
    Logger/OutputMessagesWindows.h \
    MatrixClasses/BaseFloatMatrix.h \
    MatrixClasses/BaseIndexMatrix.h \
    MatrixClasses/BaseMatrix.h \
    MatrixClasses/ComplexMatrix.h \
    MatrixClasses/CufftComplexMatrix.h \
    MatrixClasses/IndexMatrix.h \
    MatrixClasses/RealMatrix.h \
    OutputStreams/BaseOutputStream.h \
    OutputStreams/CuboidOutputStream.h \
    OutputStreams/IndexOutputStream.h \
    OutputStreams/WholeDomainOutputStream.h \
    Parameters/CommandLineParameters.h \
    Parameters/Parameters.h \
    Parameters/CudaParameters.h \
    Utils/DimensionSizes.h \
    Utils/ErrorMessages.h \
    Utils/MatrixNames.h \
    Utils/TimeMeasure.h \

OTHER_FILES += \
    KSpaceSolver/SolverCUDAKernels.cuh \
    KSpaceSolver/SolverCUDAKernels.cu \
    OutputStreams/OutputStreamsCudaKernels.cuh \
    OutputStreams/OutputStreamsCudaKernels.cu \
    Parameters/CudaDeviceConstants.cuh \
    Parameters/CudaDeviceConstants.cu \
    Utils/CudaUtils.cuh \

CUDA_SOURCES += \
    KSpaceSolver/SolverCUDAKernels.cu \
    OutputStreams/OutputStreamsCudaKernels.cu \
    Parameters/CudaDeviceConstants.cu \


NVCC_OPTIONS = --use_fast_math -Wno-deprecated-gpu-targets -rdc=true -cudart static
NVCC_OPTIONS += -Xcompiler \"/FS /EHsc /W3 /nologo /Zi\"

INCLUDEPATH += $$(CUDA_PATH)/include

QMAKE_LIBDIR += $$(CUDA_PATH)/lib/x64/

CUDA_LIBS  += -lcudart -lcufft

LIBS  += $$CUDA_LIBS cuda.obj

CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')

# MSVCRT link option (static or dynamic, it must be the same with your Qt SDK link option)
MSVCRT_LINK_FLAG_DEBUG = \"/MDd /Od /RTC1\"
MSVCRT_LINK_FLAG_RELEASE = \"/MD /O2\"

# Configuration of the Cuda compiler
CONFIG(debug, debug|release) {
    # Debug mode
    cuda_d.input = CUDA_SOURCES
    cuda_d.output = ${QMAKE_FILE_BASE}.obj
    cuda_d.commands = \"$$(CUDA_PATH)/bin/nvcc.exe\" -D_DEBUG -g $$NVCC_OPTIONS $$CUDA_INC $$CUDA_LIBS --machine 64 -Xcompiler $$MSVCRT_LINK_FLAG_DEBUG -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda_d.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda_d
} else {
    # Release mode
    cuda.input = CUDA_SOURCES
    cuda.output = ${QMAKE_FILE_BASE}.obj
    cuda.commands = \"$$(CUDA_PATH)/bin/nvcc.exe\" $$NVCC_OPTIONS $$CUDA_INC $$CUDA_LIBS --machine 64 -Xcompiler $$MSVCRT_LINK_FLAG_RELEASE -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda
}

QMAKE_PRE_LINK = \"$$(CUDA_PATH)/bin/nvcc.exe\" $$NVCC_OPTIONS -Wno-deprecated-gpu-targets -dlink $(OBJECTS) -o cuda.obj

QMAKE_POST_LINK += $${QMAKE_COPY} \"$$(CUDA_PATH)\bin\cudart*.dll\" \"$$OUT_PWD/\" &
QMAKE_POST_LINK += $${QMAKE_COPY} \"$$(CUDA_PATH)\bin\cufft*.dll\" \"$$OUT_PWD/\"

