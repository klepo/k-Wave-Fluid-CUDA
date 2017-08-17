/**
 * @file      ErrorMessages.h
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The header file containing routines for error messages and error messages common for both linux and
 *            windows version. The specific error messages are in separate files ErrorMessagesLinux.h
 *            and ErrorMessagesWindows.h
 *
 * @version   kspaceFirstOrder3D 3.5

 * @date      09 August    2011, 12:34 (created) \n
 *            16 August    2017, 13:49 (revised)
 *
 * @copyright Copyright (C) 2017 Jiri Jaros and Bradley Treeby.
 *
 * This file is part of the C++ extension of the k-Wave Toolbox [k-Wave Toolbox](http://www.k-wave.org).
 *
 * This file is part of the k-Wave. k-Wave is free software: you can redistribute it and/or modify it under the terms
 * of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * k-Wave is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for
 * more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along with k-Wave.
 * If not, see [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/).
 */


#ifndef ERROR_MESSAGES_H
#define ERROR_MESSAGES_H

#ifdef __linux__
  #include <Logger/ErrorMessagesLinux.h>
#endif

// Windows build
#ifdef _WIN64
  #include <Logger/ErrorMessagesWindows.h>
#endif

//--------------------------------------------------------------------------------------------------------------------//
//-------------------------------- Common error messages for both Linux and Windows ----------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/// delimiters for linux paths
ErrorMessage kErrFmtPathDelimiters
  = "/\\_,.:-| ()[]{}";
/// error message - out of memory
ErrorMessage  kErrFmtOutOfMemory
  = "Error: Not enough CPU or GPU memory to run this simulation.";
/// Unknown error - unknown error
ErrorMessage  kErrFmtUnknownError
  = "Error: An unknown error happened. ";

//----------------------------------------------- HDF5 error messages ------------------------------------------------//
/// HDF5 error message
ErrorMessage kErrFmtCannotCreateFile
  = "Error: File \"%s\" could not be created.";
/// HDF5 error message
ErrorMessage kErrFmtCannotRecreateFile
  = "Error: Cannot recreate an opened file \"%s\".";
/// HDF5 error message
ErrorMessage kErrFmtCannotReopenFile
  = "Error: Cannot reopen an opened file \"%s\".";
/// HDF5 error message
ErrorMessage kErrFmtCannotCloseFile
  = "Error: File \"%s\" could not be closed.";
/// HDF5 error message
ErrorMessage kErrFmtCannotWriteDataset
  = "Error: Could not write into \"%s\" dataset.";
/// HDF5 error message
ErrorMessage kErrFmtCannotReadDataset
  = "Error: Could not read from \"%s\" dataset.";
/// HDF5 error message
ErrorMessage kErrFmtBadDimensionSizes
  =  "Error: Dataset \"%s\"  has wrong dimension sizes.";
/// HDF5 error message
ErrorMessage kErrFmtFileNotOpen
  = "Error: File \"%s\" was not found or could not be opened.";
/// HDF5 error message
ErrorMessage kErrFmtNOtHdf5File
  = "Error: File \"%s\" is not a valid HDF5 file.";
/// HDF5 error message
ErrorMessage kErrFmtCannotOpenDataset
  = "Error: File \"%s\" could not open dataset \"%s\".";
/// HDF5 error message
ErrorMessage kErrFmtCannotSetCompression
  = "Error: File \"%s\", dataset \"%s\" could set compression level [%ld].";
/// HDF5 error message
ErrorMessage kErrFmtBadAttributeValue
  = "Error: Bad attribute value: [%s,%s] = %s.";
/// HDF5 error message
ErrorMessage kErrFmtCannotWriteAttribute
  = "Error: Could not write into \"%s\" attribute of \"%s\" dataset.";
/// HDF5 error message
ErrorMessage kErrFmtCannotReadAttribute
  = "Error: Could not read from \"%s\" attribute of \"%s\" dataset.";
/// HDF5 error message
ErrorMessage kErrFmtCannotCreateGroup
  = "Error: Could not create group \"%s\" in file \"%s\".";
/// HDF5 error message
ErrorMessage kErrFmtCannotOpenGroup
  = "Error: Could not open group \"%s\" in file \"%s\".";
/// HDF5 error message
ErrorMessage kErrFmtBadInputFileType
  = "Error: The input file has not a valid format.";
/// HDF5 error message
ErrorMessage kErrFmtBadOutputFIleType
  = "Error: The output file has not a valid format.";
/// HDF5 error message
ErrorMessage kErrFmtBadCheckpointFileType
  = "Error: The checkpoint file has not a valid format.";


//------------------------------------------------- Matrix Classes ---------------------------------------------------//
/// Matrix class error message
ErrorMessage  kErrFmtMatrixNotFloat
  = "Error: Matrix [%s] data type is not of single precision floating point.";
/// Matrix class error message
ErrorMessage  kErrFmtMatrixNotReal
  = "Error: Matrix [%s] domain is not real.";
/// Matrix class error message
ErrorMessage  kErrFmtMatrixNotComplex
  = "Error: Matrix [%s] domain is not complex.";
/// Matrix class error message
ErrorMessage  kErrFmtMatrixNotIndex
  = "Error: Matrix [%s] data type is not unsigned long.";


//------------------------------------------------ Matrix Container --------------------------------------------------//
/// Matrix container error message
ErrorMessage  kErrFmtBadMatrixType =
  "Error: Matrix [%s] has unknown type in the C++ code. [File, Line] : [%s,%d].";

/// Matrix container error message
ErrorMessage  kErrFmtRelocationError =
  "Error: Matrix [%s] is being reallocated in matrix container.";


//-------------------------------------------- Command line Parameters -----------------------------------------------//
/// Command line parameters error message
ErrorMessage kErrFmtNoProgressPrintInterval
  = "Error: No or invalid progress print interval.";
/// Command line parameters error message
ErrorMessage kErrFmtInvalidNumberOfThreads
  = "Error: No or invalid number of CPU threads.";
/// Command line parameters error message
ErrorMessage kErrFmtNoDeviceIndex
  = "Error: No or invalid id of the GPU device.";
/// Command line parameters error message
ErrorMessage kErrFmtNoCompressionLevel
  = "Error: No or invalid compression level.";
/// Command line parameters error message
ErrorMessage kErrFmtNoSamplingStartTimeStep
  = "Error: No or invalid collection start time step.";
/// Command line parameters error message
ErrorMessage kErrFmtNoBenchmarkTimeStep
  = "Error: No or invalid number of time step to benchmark.";
/// Command line parameters error message
ErrorMessage kErrFmtNoVerboseLevel
  = "Error: No or invalid verbose level.";

/// Error message - input file was not specified
ErrorMessage kErrFmtNoInputFile
  = "Error: The input file was not specified.";
/// Command line parameters error message
ErrorMessage kErrFmtNoOutputFile
  = "Error: The output file was not specified.";
/// Command line parameters error message
ErrorMessage kErrFmtNoCheckpointFile
  = "Error: The checkpoint file was not specified.";
/// Command line parameters error message
ErrorMessage kErrFmtNoCheckpointInterval
  = "Error: The checkpoint interval was not specified.";
/// Command line parameter error message
ErrorMessage kErrFmtUnknownParameter
  = "Error: Unknown command line parameter.";
/// Command line parameter error message
ErrorMessage kErrFmtUnknownParameterOrArgument
  = "Error: Unknown command line parameter or missing argument.";

/// Command line parameters error message
ErrorMessage kErrFmtIllegalAlphaPowerValue
  = "Error: Illegal value of alpha_power (must not equal to 1.0).";
/// Command line parameters error message
ErrorMessage kErrFmtIllegalSamplingStartTimeStep
  = "Error: The beginning of data sampling is out of the simulation time span <%zu, %zu>.";

/// Command line parameters error message
ErrorMessage kErrFmtBadInputFileFormat
  = "Error: Incorrect input file\"%s\" format.";
/// Command line parameters error message
ErrorMessage kErrFmtBadMajorFileVersion
  = "Error: Incorrect major version of the HDF5 file %s (expected is %s).";
/// Command line parameters error message
ErrorMessage kErrFmtBadMinorFileVersion
  = "Error: Incorrect minor version of the HDF5 file %s (expected is %s).";
/// Command line parameters error message
ErrorMessage kErrFmtBadSensorMaskType
  = "Error: The sensor mask type specified in the input file is not supported.";
/// Command line parameters error message
ErrorMessage kErrFmtNonStaggeredVelocityNotSupportedFileVersion
  = "Error: --u_non_staggered_raw is not supported along with the input file of the version 1.0.";


//-------------------------------------------- KSpaceFirstOrder3DSolver ----------------------------------------------//
/// KSpaceFirstOrder3DSolver error message
ErrorMessage kErrFmtBadCheckpointFileFormat
  = "Error: Incorrect checkpoint file \"%s\" format.";

/// KSpaceFirstOrder3DSolver error message
ErrorMessage kErrFmtBadOutputFileFormat
  = "Error: Incorrect output file \"%s\" format.";

/// KSpaceFirstOrder3DSolver error message
ErrorMessage kErrFmtCheckpointDimensionsNotMatch
  = "Error: The dimensions [%ld, %ld, %ld] of the checkpoint file don't match the simulation "
    "dimensions [%ld, %ld, %ld].";

/// KSpaceFirstOrder3DSolver error message
ErrorMessage kErrFmtOutputDimensionsNotMatch
  = "Error: The dimensions [%ld, %ld, %ld] of the output file don't match the simulation "
    "dimensions [%ld, %ld, %ld].";



//------------------------------------------------ Cuda fft errors ---------------------------------------------------//
/// CUDA FFT error message.
ErrorMessage kErrFmtCufftInvalidPlan
  = "Error: cuFFT was passed an invalid plan handle while %s.";
/// CUDA FFT error message.
ErrorMessage kErrFmtCufftAllocFailed
  = "Error: cuFFT failed to allocate GPU or CPU memory while %s.";
/// CUDA FFT error message.
ErrorMessage kErrFmtCufftInvalidType
  = "Error: cuFFT invalid type for of the transform while %s.";
/// CUDA FFT error message.
ErrorMessage kErrFmtCufftInvalidValue
  = "Error: cuFFT was given an invalid pointer or parameter while %s.";
/// CUDA FFT error message.
ErrorMessage kErrFmtCuFFTInternalError
  = "Error: Driver or internal cuFFT library error while %s.";
/// CUDA FFT error message.
ErrorMessage kErrFmtCufftExecFailed
  = "Error: Failed to execute an cuFFT on the GPU while %s.";
/// CUDA FFT error message.
ErrorMessage kErrFmtCufftSetupFailed
  = "Error: The cuFFT library failed to initialize while %s.";
/// CUDA FFT error message.
ErrorMessage kErrFmtCufftInvalidSize
  = "Error: cuFFT was given an invalid transform size while %s.";
/// CUDA FFT error message.
ErrorMessage kErrFmtCufftUnalignedData
  = "Error: Arrays for cuFFT was not properly aligned while %s.";
/// CUDA FFT error message.
ErrorMessage kErrFmtCufftIncompleteParaterList
  = "Error: Missing parameters in the cuFFT call while %s.";
/// CUDA FFT error message.
ErrorMessage kErrFmtCufftInvalidDevice
  = "Error: cuFFT execution of the plan was performed on a different GPU than plan was "
    "created while %s.";
/// CUDA FFT error message.
ErrorMessage kErrFmtCufftParseError
  = "Error: cuFFT internal plan database error while %s.";
/// CUDA FFT error message.
ErrorMessage kErrFmtCufftNoWorkspace
  = "Error: No workspace has been provided prior to cuFFT plan execution while %s.";
/// CUDA FFT error message.
ErrorMessage kErrFmtCufftNotImplemented
  = "Error: cuFFT feature is not implemented while %s.";
/// CUDA FFT error message.
ErrorMessage kErrFmtCufftLicenseError
  = "Error: cuFFT license error while %s.";
/// CUDA FFT error message.
ErrorMessage kErrFmtCufftNotSupported
  = "Error: cuFFT operation is not supported for parameters given while %s.";
/// CUDA FFT error message.
ErrorMessage kErrFmtCufftUnknownError
  = "Error: cuFFT failed with unknown error while %s.";

/// CUDA FFT error message.
ErrorMessage kErrFmtCreateR2CFftPlan3D
  = "creating plan for 3D real-to-complex fft";
/// CUDA FFT error message.
ErrorMessage kErrFmtCreateC2RFftPlan3D
 = "creating plan for 3D complex-to-real fft";
/// CUDA FFT error message.
ErrorMessage kErrFmtCreateR2CFftPlan1DX
 = "creating for 1D real-to-complex fft plan in X direction";
/// CUDA FFT error message.
ErrorMessage kErrFmtCreateR2CFftPlan1DY
 = "creating for 1D real-to-complex fft plan in Y direction";
/// CUDA FFT error message.
ErrorMessage kErrFmtCreateR2CFftPlan1DZ
 = "creating for 1D real-to-complex fft plan in Z direction";
/// CUDA FFT error message.
ErrorMessage kErrFmtCreateC2RFftPlan1DX
  = "creating for 1D complex-to-real fft plan in X direction";
/// CUDA FFT error message.
ErrorMessage kErrFmtCreateC2RFftPlan1DY
  = "creating for 1D complex-to-real fft plan in Y direction";
/// CUDA FFT error message.
ErrorMessage kErrFmtCreateC2RFftPlan1DZ
  = "creating 1D complex-to-real fft plan in Z direction";

/// CUDA FFT error message.
ErrorMessage kErrFmtDestroyR2CFftPlan3D
  = "destroying plan for 3D real-to-complex fft";
/// CUDA FFT error message.
ErrorMessage kErrFmtDestroyC2RFftPlan3D
 = "destroying plan for 3D complex-to-real fft";
/// CUDA FFT error message.
ErrorMessage kErrFmtDestroyR2CFftPlan1DX
 = "destroying for 1D real-to-complex fft plan in X direction";
/// CUDA FFT error message.
ErrorMessage kErrFmtDestroyR2CFftPlan1DY
 = "destroying for 1D real-to-complex fft plan in Y direction";
/// CUDA FFT error message.
ErrorMessage kErrFmtDestroyR2CFftPlan1DZ
 = "destroying for 1D real-to-complex fft plan in Z direction";
/// CUDA FFT error message.
ErrorMessage kErrFmtDestroyC2RFftPlan1DX
  = "destroying for 1D complex-to-real fft plan in X direction";
/// CUDA FFT error message.
ErrorMessage kErrFmtDestroyC2RFftPlan1DY
  = "destroying for 1D complex-to-real fft plan in Y direction";
/// CUDA FFT error message.
ErrorMessage kErrFmtDestroyC2RFftPlan1DZ
  = "destroying 1D complex-to-real fft plan in Z direction";

/// CUDA FFT error message.
ErrorMessage kErrFmtExecuteR2CFftPlan3D
  = "executing plan for 3D real-to-complex fft";
/// CUDA FFT error message.
ErrorMessage kErrFmtExecuteC2RFftPlan3D
 = "executing plan for 3D complex-to-real fft";
/// CUDA FFT error message.
ErrorMessage kErrFmtExecuteR2CFftPlan1DX
 = "executing for 1D real-to-complex fft plan in X direction";
/// CUDA FFT error message.
ErrorMessage kErrFmtExecuteR2CFftPlan1DY
 = "executing for 1D real-to-complex fft plan in Y direction";
/// CUDA FFT error message.
ErrorMessage kErrFmtExecuteR2CFftPlan1DZ
 = "executing for 1D real-to-complex fft plan in Z direction";
/// CUDA FFT error message.
ErrorMessage kErrFmtExecuteC2RFftPlan1DX
  = "executing for 1D complex-to-real fft plan in X direction";
/// CUDA FFT error message.
ErrorMessage kErrFmtExecuteC2RFftPlan1DY
  = "executing for 1D complex-to-real fft plan in Y direction";
/// CUDA FFT error message.
ErrorMessage kErrFmtExecuteC2RFftPlan1DZ
  = "executing 1D complex-to-real fft plan in Z direction";


//---------------------------------------------- CudaParameters Class ------------------------------------------------//
/// CUDATuner error message
ErrorMessage kErrFmtBadDeviceIndex
  = "Error: Wrong CUDA device id %d. Allowed devices <0, %d>.";
/// CUDATuner error message
ErrorMessage kErrFmtNoFreeDevice
  = "Error: All CUDA-capable devices are busy or unavailable.";
/// CUDATuner error message
ErrorMessage kErrFmtDeviceIsBusy
  = "Error: CUDA device id %d is busy or unavailable.";

/// CUDAParameters error message
ErrorMessage kErrFmtInsufficientCudaDriver
  = "Error: Insufficient CUDA driver version. The code needs CUDA version "
    "%d.%d but %d.%d is installed.";
/// CUDAParameters error message
ErrorMessage kErrFmtCannotReadCudaVersion
  = "Error: Insufficient CUDA driver version. Install the latest drivers.";
/// CUDAParameters error message
ErrorMessage kErrFmtDeviceNotSupported
  = "Error: CUDA device id %d is not supported by this k-Wave build.";


//----------------------------------------------- CheckErrors header -------------------------------------------------//
/// CUDAParameters error message
ErrorMessage kErrFmtDeviceError
  = "GPU error: %s routine name: %s in file %s, line %d.";


#endif	/* ERROR_MESSAGES_H */
