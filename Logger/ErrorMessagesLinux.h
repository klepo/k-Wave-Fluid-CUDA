/**
 * @file        ErrorMessagesLinux.h
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file containing all error messages of the project
 *              and routines to handle errors (for CUDA), Linux version.
 *
 * @version     kspaceFirstOrder3D 3.4
 * @date        13 July     2016, 11:26 (created) \n
 *              14 July     2016, 13:20 (revised)
 *
 * @section License
 * This file is part of the C++ extension of the k-Wave Toolbox
 * (http://www.k-wave.org).\n Copyright (C) 2014 Jiri Jaros, Beau Johnston
 * and Bradley Treeby
 *
 * This file is part of the k-Wave. k-Wave is free software: you can
 * redistribute it and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation, either version
 * 3 of the License, or (at your option) any later version.
 *
 * k-Wave is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for
 * more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with k-Wave. If not, see http://www.gnu.org/licenses/.
 */


#ifndef ERROR_MESSAGES_LINUX_H
#define	ERROR_MESSAGES_LINUX_H

//----------------------------- HDF5 error messages --------------------------//
/**
 * @typedef TErrorMessage
 * @brief Datatype for error messages
 */
typedef const char * const TErrorMessage;

/// Error message header
TErrorMessage ERR_FMT_HEAD =
        "┌───────────────────────────────────────────────────────────────┐\n"
        "│            !!! K-Wave experienced a fatal error !!!           │\n"
        "├───────────────────────────────────────────────────────────────┤\n";

        /// Error message tailer
TErrorMessage ERR_FMT_TAIL =
        "├───────────────────────────────────────────────────────────────┤\n"
        "│                      Execution terminated                     │\n"
        "└───────────────────────────────────────────────────────────────┘\n";

/// delimiters for linux paths
TErrorMessage ERR_FMTPathDelimiters = "/\\_,.:-| ()[]{}";

/// error message
TErrorMessage  ERR_FMT_Not_Enough_Memory = "Error: Not enough CPU or GPU memory to run this simulation.";
/// Unknown error
TErrorMessage  ERR_FMT_UnknownError = "Error: An unknown error happened. ";


/// HDF5 error message
TErrorMessage HDF5_ERR_FMT_FileNotCreated          = "Error: File \"%s\" could not be created.";
/// HDF5 error message
TErrorMessage HDF5_ERR_FMT_FileCannotRecreated     = "Error: Cannot recreate an opened file \"%s\".";
/// HDF5 error message
TErrorMessage HDF5_ERR_FMT_FileCannotReopen        = "Error: Cannot reopen an opened file \"%s\".";
/// HDF5 error message
TErrorMessage HDF5_ERR_FMT_FileNotClosed           = "Error: File \"%s\" could not be closed.";
/// HDF5 error message
TErrorMessage HDF5_ERR_FMT_CouldNotWriteTo         = "Error: Could not write into \"%s\" dataset.";
/// HDF5 error message
TErrorMessage HDF5_ERR_FMT_CouldNotReadFrom        = "Error: Could not read from \"%s\" dataset.";
/// HDF5 error message
TErrorMessage HDF5_ERR_FMT_WrongDimensionSizes     = "Error: Dataset \"%s\"  has wrong dimension sizes.";
/// HDF5 error message
TErrorMessage HDF5_ERR_FMT_FileNotOpened           = "Error: File \"%s\" was not found or could not be opened.";
/// HDF5 error message
TErrorMessage HDF5_ERR_FMT_NotHDF5File             = "Error: File \"%s\" is not a valid HDF5 file.";
/// HDF5 error message
TErrorMessage HDF5_ERR_FMT_DatasetNotOpened        = "Error: File \"%s\" could not open dataset \"%s\".";
/// HDF5 error message
TErrorMessage HDF5_ERR_FMT_CouldNotSetCompression  = "Error: File \"%s\", dataset \"%s\" could set compression level [%ld].";
/// HDF5 error message
TErrorMessage HDF5_ERR_FMT_BadAttributeValue       = "Error: Bad attribute value: [%s,%s] = %s.";
/// HDF5 error message
TErrorMessage HDF5_ERR_FMT_CouldNotWriteToAttribute  = "Error: Could not write into \"%s\" attribute of \"%s\" dataset.";
/// HDF5 error message
TErrorMessage HDF5_ERR_FMT_CouldNotReadFromAttribute = "Error: Could not read from \"%s\" attribute of \"%s\" dataset.";
/// HDF5 error message
TErrorMessage HDF5_ERR_FMT_GroupNotCreated          = "Error: Could not create group \"%s\" in file \"%s\".";
/// HDF5 error message
TErrorMessage HDF5_ERR_FMT_GroupNotOpened           = "Error: Could not open group \"%s\" in file \"%s\".";
/// HDF5 error message
TErrorMessage HDF5_ERR_FMT_BadInputFileType         = "Error: The input file has not a valid format.";
/// HDF5 error message
TErrorMessage HDF5_ERR_FMT_BadOutputFileType        = "Error: The output file has not a valid format.";
/// HDF5 error message
TErrorMessage HDF5_ERR_FMT_BadCheckpointFileType    = "Error: The checkpoint file has not a valid format.";

//---------------------------------- Matrix Classes  -------------------------//


/// Matrix class error message
TErrorMessage  Matrix_ERR_FMT_MatrixNotFloat     = "Error: Matrix [%s] data type is not of single precision floating point.";
/// Matrix class error message
TErrorMessage  Matrix_ERR_FMT_MatrixNotReal      = "Error: Matrix [%s] domain is not real.";
/// Matrix class error message
TErrorMessage  Matrix_ERR_FMT_MatrixNotComplex   = "Error: Matrix [%s] domain is not complex.";
/// Matrix class error message
TErrorMessage  Matrix_ERR_FMT_MatrixNotLong      = "Error: Matrix [%s] data type is not unsigned long.";

//--------------------------------- Matrix Container  ------------------------//

/// Matrix container error message
TErrorMessage  MatrixContainer_ERR_FMT_RecordUnknownDistributionType =
        "Error: Matrix [%s] has unknown distribution type in the C++ code. "
        "[File, Line] : [%s,%d].";

/// Matrix container error message
TErrorMessage  MatrixContainer_ERR_FMT_ReloactaionError =
        "Error: Matrix [%s] is being reallocated. [File, line] : [%s,%d].";


//-------------------------- Command line Parameters  ------------------------//

/// Command line parameters error message
TErrorMessage ERR_FMT_NoProgressPrintIntreval  = "Error: No or invalid progress print interval.";
/// Command line parameters error message
TErrorMessage ERR_FMT_NoThreadNumbers          = "Error: No or invalid number of CPU threads.";
/// Command line parameters error message
TErrorMessage ERR_FMT_NoGPUNumber              = "Error: No or invalid id of the GPU device.";
/// Command line parameters error message
TErrorMessage ERR_FMT_NoCompressionLevel       = "Error: No or invalid compression level.";
/// Command line parameters error message
TErrorMessage ERR_FMT_NoStartTimestep          = "Error: No or invalid collection start time step.";
/// Command line parameters error message
TErrorMessage ERR_FMT_NoBenchmarkTimeStepCount = "Error: No or invalid number of time step to benchmark.";
/// Command line parameters error message
TErrorMessage ERR_FMT_BadVerboseLevel          = "Error: No or invalid verbose level.";

/// Error message - input file was not specified
TErrorMessage ERR_FMT_NoInputFile          = "Error: The input file was not specified.";
/// Command line parameters error message
TErrorMessage ERR_FMT_NoOutputFile         = "Error: The output file was not specified.";
/// Command line parameters error message
TErrorMessage ERR_FMT_NoCheckpointFile     = "Error: The checkpoint file was not specified.";
/// Command line parameters error message
TErrorMessage ERR_FMT_NoCheckpointInterval = "Error: The checkpoint interval was not specified.";
/// Command line parameter error message
TErrorMessage ERR_FMT_UnknownParameter     = "Error: Unknown command line parameter.";
/// Command line parameter error message
TErrorMessage ERR_FMT_UnknownParameterOrMissingArgument = "Error: Unknown command line parameter or missing argument.";

/// Command line parameters error message
TErrorMessage Parameters_ERR_FMT_Illegal_alpha_power_value = "Error: Illegal value of alpha_power (must not equal to 1.0).";
/// Command line parameters error message
TErrorMessage Parameters_ERR_FMT_Illegal_StartTime_value   = "Error: The beginning of data sampling is out of the simulation time span <%zu, %zu>.";

/// Command line parameters error message
TErrorMessage Parameters_ERR_FMT_IncorrectInputFileFormat = "Error: Incorrect input file\"%s\" format.";
/// Command line parameters error message
TErrorMessage Parameters_ERR_FMT_IncorrectMajorHDF5FileVersion = "Error: Incorrect major version of the HDF5 file %s (expected is %s).";
/// Command line parameters error message
TErrorMessage Parameters_ERR_FMT_IncorrectMinorHDF5FileVersion = "Error: Incorrect minor version of the HDF5 file %s (expected is %s).";
/// Command line parameters error message
TErrorMessage Parameters_ERR_FMT_WrongSensorMaskType = "Error: The sensor mask type specified in the input file is not supported.";
/// Command line parameters error message
TErrorMessage Parameters_ERR_FMT_UNonStaggeredNotSupportedForFile10 = "Error: --u_non_staggered_raw is not supported along with the input file of the version 1.0.";



//------------------------------- CUDA  --------------------------------------//

//Zde
//-------------------------------- CUDA FFT Errors ---------------------------//
/// CUDA FFT error message.
TErrorMessage CUFFTComplexMatrix_ERR_FMT_INVALID_PLAN         = "Error: cuFFT was passed an invalid plan handle for %s.";
/// CUDA FFT error message.
TErrorMessage CUFFTComplexMatrix_ERR_FMT_CUFFT_ALLOC_FAILED   = "Error: cuFFT failed to allocate GPU or CPU memory for %s.";
/// CUDA FFT error message.
TErrorMessage CUFFTComplexMatrix_ERR_FMT_CUFFT_INVALID_TYPE   = "Error: cuFFT invalid type for of the transform for %s.";
/// CUDA FFT error message.
TErrorMessage CUFFTComplexMatrix_ERR_FMT_CUFFT_INVALID_VALUE  = "Error: cuFFT was given an invalid pointer or parameter for %s.";
/// CUDA FFT error message.
TErrorMessage CUFFTComplexMatrix_ERR_FMT_CUFFT_INTERNAL_ERROR = "Error: Driver or internal cuFFT library error for %s.";
/// CUDA FFT error message.
TErrorMessage CUFFTComplexMatrix_ERR_FMT_CUFFT_EXEC_FAILED    = "Error: Failed to execute an cuFFT on the GPU for %s.";
/// CUDA FFT error message.
TErrorMessage CUFFTComplexMatrix_ERR_FMT_CUFFT_SETUP_FAILED   = "Error: The cuFFT library failed to initialize for %s.";
/// CUDA FFT error message.
TErrorMessage CUFFTComplexMatrix_ERR_FMT_CUFFT_INVALID_SIZE   = "Error: cuFFT was given an invalid transform size for %s.";
/// CUDA FFT error message.
TErrorMessage CUFFTComplexMatrix_ERR_FMT_CUFFT_UNALIGNED_DATA = "Error: Arrays for cuFFT was not properly aligned for %s.";
/// CUDA FFT error message.
TErrorMessage CUFFTComplexMatrix_ERR_FMT_CUFFT_INCOMPLETE_PARAMETER_LIST = "Error: Missing parameters in the cuFFT call for %s.";
/// CUDA FFT error message.
TErrorMessage CUFFTComplexMatrix_ERR_FMT_CUFFT_INVALID_DEVICE  = "Error: cuFFT execution of the plan was performed on a different GPU than plan was created for %s.";
/// CUDA FFT error message.
TErrorMessage CUFFTComplexMatrix_ERR_FMT_CUFFT_PARSE_ERROR     = "Error: cuFFT internal plan database error for %s.";
/// CUDA FFT error message.
TErrorMessage CUFFTComplexMatrix_ERR_FMT_CUFFT_NO_WORKSPACE    = "Error: No workspace has been provided prior to cuFFT plan execution for %s.";
/// CUDA FFT error message.
TErrorMessage CUFFTComplexMatrix_ERR_FMT_CUFFT_NOT_IMPLEMENTED = "Error: cuFFT feature is not implemented for %s.";
/// CUDA FFT error message.
TErrorMessage CUFFTComplexMatrix_ERR_FMT_CUFFT_LICENSE_ERROR   = "Error: cuFFT license error for %s.";
/// CUDA FFT error message.
TErrorMessage CUFFTComplexMatrix_ERR_FMT_CUFFT_UNKNOWN_ERROR   = "Error: cuFFT failed with unknown error for %s.";


//------------------------- KSpaceFirstOrder3DSolver Classes  ----------------//

/// KSpaceFirstOrder3DSolver error message
TErrorMessage KSpaceFirstOrder3DSolver_ERR_FMT_IncorrectCheckpointFileFormat
  = "Error: Incorrect checkpoint file \"%s\" format.";

/// KSpaceFirstOrder3DSolver error message
TErrorMessage KSpaceFirstOrder3DSolver_ERR_FMT_IncorrectOutputFileFormat
  = "Error: Incorrect output file \"%s\" format.";

/// KSpaceFirstOrder3DSolver error message
TErrorMessage KSpaceFirstOrder3DSolver_ERR_FMT_CheckpointDimensionsDoNotMatch
  = "Error: The dimensions [%ld, %ld, %ld] of the checkpoint file don't match the simulation dimensions [%ld, %ld, %ld].";

/// KSpaceFirstOrder3DSolver error message
TErrorMessage KSpaceFirstOrder3DSolver_ERR_FMT_OutputDimensionsDoNotMatch
  = "Error: The dimensions [%ld, %ld, %ld] of the output file don't match the simulation dimensions [%ld, %ld, %ld].";


//------------------------------ CUDAParameters Class  -----------------------//
/// CUDATuner error message
TErrorMessage CUDAParameters_ERR_FMT_WrongDeviceIdx    = "Error: Wrong CUDA device id %d. Allowed devices <0, %d>.";
/// CUDATuner error message
TErrorMessage CUDAParameters_ERR_FMT_NoFreeDevice      = "Error: All CUDA-capable devices are busy or unavailable.";
/// CUDATuner error message
TErrorMessage CUDAParameters_ERR_FMT_DeviceIsBusy      = "Error: CUDA device id %d is busy or unavailable.";

/// CUDAParameters error message
TErrorMessage CUDAParameters_ERR_FMT_InsufficientCUDADriver = "Error: Insufficient CUDA driver version. The code needs CUDA version %d.%d but %d.%d is installed.";
/// CUDAParameters error message
TErrorMessage CUDAParameters_ERR_FM_CannotReadCUDAVersion   = "Error: Insufficient CUDA driver version. Install the latest drivers.";
/// CUDAParameters error message
TErrorMessage CUDAParameters_ERR_FM_GPUNotSupported         = "Error: CUDA device id %d is not supported by this k-Wave build.";


//------------------------------ CheckErrors header --------------------------//
TErrorMessage CUDACheckErrors_ERR_FM_GPU_Error = "GPU error: %s routine name: %s in file %s, line %d.";

#endif	/* ERROR_MESSAGES_LINUX_H */