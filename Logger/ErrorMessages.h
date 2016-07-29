/**
 * @file        ErrorMessages.h
 *
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file containing routines for error messages and error messages common for
 *              both linux and windows version. The speficic error messages are in separate files
 *              ErrorMessagesLinux.h and ErrorMessagesWindows.h
 *
 * @version     kspaceFirstOrder3D 3.4
 *
 * @date        09 August   2011, 12:34 (created) \n
 *              29 July     2016, 16:42 (revised)
 *
  * @section License
 * This file is part of the C++ extension of the k-Wave Toolbox
 * (http://www.k-wave.org).\n Copyright (C) 2016 Jiri Jaros and Bradley Treeby.
 *
 * This file is part of the k-Wave. k-Wave is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later version.
 *
 * k-Wave is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser
 * General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along with k-Wave.
 * If not, see http://www.gnu.org/licenses/.
 */


#ifndef ERROR_MESSAGES_H
#define	ERROR_MESSAGES_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

#ifdef __linux__
  #include <Logger/ErrorMessagesLinux.h>
#endif

// Windows build
#ifdef _WIN64
  #include <Logger/ErrorMessagesWindows.h>
#endif

//------------------------------------------------------------------------------------------------//
//---------------------- Common error messages for both Linux and Windows ------------------------//
//------------------------------------------------------------------------------------------------//

/// delimiters for linux paths
TErrorMessage ERR_FMT_PATH_DELIMITERS = "/\\_,.:-| ()[]{}";

/// error message - out of memory
TErrorMessage  ERR_FMT_OUT_OF_MEMORY =
        "Error: Not enough CPU or GPU memory to run this simulation.";
/// Unknown error - unknown error
TErrorMessage  ERR_FMT_UNKNOWN_ERROR
        = "Error: An unknown error happened. ";

//------------------------------------- HDF5 error messages --------------------------------------//

/// HDF5 error message
TErrorMessage ERR_FMT_CANNOT_CREATE_FILE
        = "Error: File \"%s\" could not be created.";
/// HDF5 error message
TErrorMessage ERR_FMT_CANNOT_RECREATE_FILE
        = "Error: Cannot recreate an opened file \"%s\".";
/// HDF5 error message
TErrorMessage ERR_FMT_CANNOT_REOPEN_FILE
        = "Error: Cannot reopen an opened file \"%s\".";
/// HDF5 error message
TErrorMessage ERR_FMT_CANNOT_CLOSE_FILE
        = "Error: File \"%s\" could not be closed.";
/// HDF5 error message
TErrorMessage ERR_FMT_CANNOT_WRITE_DATASET
        = "Error: Could not write into \"%s\" dataset.";
/// HDF5 error message
TErrorMessage ERR_FMT_CANNOT_READ_DATASET
        = "Error: Could not read from \"%s\" dataset.";
/// HDF5 error message
TErrorMessage ERR_FMT_BAD_DIMENSION_SIZES
        =  "Error: Dataset \"%s\"  has wrong dimension sizes.";
/// HDF5 error message
TErrorMessage ERR_FMT_FILE_NOT_OPEN
        = "Error: File \"%s\" was not found or could not be opened.";
/// HDF5 error message
TErrorMessage ERR_FMT_NOT_HDF5_FILE
        = "Error: File \"%s\" is not a valid HDF5 file.";
/// HDF5 error message
TErrorMessage ERR_FMT_CANNOT_OPEN_DATASET
        = "Error: File \"%s\" could not open dataset \"%s\".";
/// HDF5 error message
TErrorMessage ERR_FMT_CANNOT_SET_COMPRESSION
        = "Error: File \"%s\", dataset \"%s\" could set compression level [%ld].";
/// HDF5 error message
TErrorMessage ERR_FMT_BAD_ATTRIBUTE_VALUE
        = "Error: Bad attribute value: [%s,%s] = %s.";
/// HDF5 error message
TErrorMessage ERR_FMT_CANNOT_WRITE_ATTRIBUTE
        = "Error: Could not write into \"%s\" attribute of \"%s\" dataset.";
/// HDF5 error message
TErrorMessage ERR_FMT_CANNOT_READ_ATTRIBUTE
        = "Error: Could not read from \"%s\" attribute of \"%s\" dataset.";
/// HDF5 error message
TErrorMessage ERR_FMT_CANNOT_CREATE_GROUP
        = "Error: Could not create group \"%s\" in file \"%s\".";
/// HDF5 error message
TErrorMessage ERR_FMT_CANNOT_OPEN_GROUP
        = "Error: Could not open group \"%s\" in file \"%s\".";
/// HDF5 error message
TErrorMessage ERR_FMT_BAD_INPUT_FILE_TYPE
        = "Error: The input file has not a valid format.";
/// HDF5 error message
TErrorMessage ERR_FMT_BAD_OUTPUT_FILE_TYPE
        = "Error: The output file has not a valid format.";
/// HDF5 error message
TErrorMessage ERR_FMT_BAD_CHECKPOINT_FILE_TYPE
        = "Error: The checkpoint file has not a valid format.";


//--------------------------------------- Matrix Classes  ----------------------------------------//

/// Matrix class error message
TErrorMessage  ERR_FMT_MATRIX_NOT_FLOAT
        = "Error: Matrix [%s] data type is not of single precision floating point.";
/// Matrix class error message
TErrorMessage  ERR_FMT_MATRIX_NOT_REAL
        = "Error: Matrix [%s] domain is not real.";
/// Matrix class error message
TErrorMessage  ERR_FMT_MATRIX_NOT_COMPLEX
        = "Error: Matrix [%s] domain is not complex.";
/// Matrix class error message
TErrorMessage  ERR_FMT_MATRIX_NOT_INDEX
        = "Error: Matrix [%s] data type is not unsigned long.";


//-------------------------------------- Matrix Container ----------------------------------------//

/// Matrix container error message
TErrorMessage  ERR_FMT_BAD_MATRIX_DISTRIBUTION_TYPE =
        "Error: Matrix [%s] has unknown distribution type in the C++ code. "
        "[File, Line] : [%s,%d].";

/// Matrix container error message
TErrorMessage  ERR_FMT_RELOCATION_ERROR =
        "Error: Matrix [%s] is being reallocated in matrix container.";


//---------------------------------- Command line Parameters  ------------------------------------//

/// Command line parameters error message
TErrorMessage FMT_NO_PROGRESS_PRINT_INTERVAL
        = "Error: No or invalid progress print interval.";
/// Command line parameters error message
TErrorMessage ERR_FMT_NO_THREAD_NUMBER
        = "Error: No or invalid number of CPU threads.";
/// Command line parameters error message
TErrorMessage ERR_FMT_NO_GPU_NUMBER
        = "Error: No or invalid id of the GPU device.";
/// Command line parameters error message
TErrorMessage ERR_FMT_NO_COMPRESSION_LEVEL
        = "Error: No or invalid compression level.";
/// Command line parameters error message
TErrorMessage ERR_FMT_NO_START_TIME_STEP
        = "Error: No or invalid collection start time step.";
/// Command line parameters error message
TErrorMessage ERR_FMT_NO_BENCHMARK_STEP_SET
        = "Error: No or invalid number of time step to benchmark.";
/// Command line parameters error message
TErrorMessage ERR_FMT_NO_VERBOSE_LEVEL
        = "Error: No or invalid verbose level.";

/// Error message - input file was not specified
TErrorMessage ERR_FMT_NO_INPUT_FILE
        = "Error: The input file was not specified.";
/// Command line parameters error message
TErrorMessage ERR_FMT_NO_OUTPUT_FILE
        = "Error: The output file was not specified.";
/// Command line parameters error message
TErrorMessage ERR_FMT_NO_CHECKPOINT_FILE
        = "Error: The checkpoint file was not specified.";
/// Command line parameters error message
TErrorMessage ERR_FMT_NO_CHECKPOINT_INTERVAL
        = "Error: The checkpoint interval was not specified.";
/// Command line parameter error message
TErrorMessage ERR_FMT_UNKNOWN_PARAMETER
        = "Error: Unknown command line parameter.";
/// Command line parameter error message
TErrorMessage ERR_FMT_UNKNOW_PARAMETER_OR_ARGUMENT
        = "Error: Unknown command line parameter or missing argument.";

/// Command line parameters error message
TErrorMessage ERR_FMT_ILLEGAL_ALPHA_POWER_VALUE
        = "Error: Illegal value of alpha_power (must not equal to 1.0).";
/// Command line parameters error message
TErrorMessage ERR_FMT_ILLEGAL_START_TIME_VALUE
        = "Error: The beginning of data sampling is out of the simulation time span <%zu, %zu>.";

/// Command line parameters error message
TErrorMessage ERR_FMT_BAD_INPUT_FILE_FORMAT
        = "Error: Incorrect input file\"%s\" format.";
/// Command line parameters error message
TErrorMessage ERR_FMT_BAD_MAJOR_File_Version
        = "Error: Incorrect major version of the HDF5 file %s (expected is %s).";
/// Command line parameters error message
TErrorMessage ERR_FMT_BAD_MINOR_FILE_VERSION
        = "Error: Incorrect minor version of the HDF5 file %s (expected is %s).";
/// Command line parameters error message
TErrorMessage ERR_FMT_BAD_SENSOR_MASK_TYPE
        = "Error: The sensor mask type specified in the input file is not supported.";
/// Command line parameters error message
TErrorMessage ERR_FMT_U_NON_STAGGERED_NOT_SUPPORTED_FILE_VERSION
        = "Error: --u_non_staggered_raw is not supported along with the input file of the version 1.0.";


//---------------------------------- KSpaceFirstOrder3DSolver ------------------------------------//

/// KSpaceFirstOrder3DSolver error message
TErrorMessage ERR_FMT_BAD_CHECKPOINT_FILE_FORMAT
        = "Error: Incorrect checkpoint file \"%s\" format.";

/// KSpaceFirstOrder3DSolver error message
TErrorMessage ERR_FMT_BAD_OUTPUT_FILE_FORMAT
        = "Error: Incorrect output file \"%s\" format.";

/// KSpaceFirstOrder3DSolver error message
TErrorMessage ERR_FMT_CHECKPOINT_DIMENSIONS_NOT_MATCH
        = "Error: The dimensions [%ld, %ld, %ld] of the checkpoint file don't match the simulation "
          "dimensions [%ld, %ld, %ld].";

/// KSpaceFirstOrder3DSolver error message
TErrorMessage ERR_FMT_OUTPUT_DIMENSIONS_NOT_MATCH
        = "Error: The dimensions [%ld, %ld, %ld] of the output file don't match the simulation "
          "dimensions [%ld, %ld, %ld].";



//-------------------------------------- CUDA FFT Errors -----------------------------------------//
/// CUDA FFT error message.
TErrorMessage ERR_FMT_CUFFT_INVALID_PLAN
        = "Error: cuFFT was passed an invalid plan handle for %s.";
/// CUDA FFT error message.
TErrorMessage ERR_FMT_CUFFT_ALLOC_FAILED
        = "Error: cuFFT failed to allocate GPU or CPU memory for %s.";
/// CUDA FFT error message.
TErrorMessage ERR_FMT_CUFFT_INVALID_TYPE
        = "Error: cuFFT invalid type for of the transform for %s.";
/// CUDA FFT error message.
TErrorMessage ERR_FMT_CUFFT_INVALID_VALUE
        = "Error: cuFFT was given an invalid pointer or parameter for %s.";
/// CUDA FFT error message.
TErrorMessage ERR_FMT_CUFFT_INTERNAL_ERROR
        = "Error: Driver or internal cuFFT library error for %s.";
/// CUDA FFT error message.
TErrorMessage ERR_FMT_CUFFT_EXEC_FAILED
        = "Error: Failed to execute an cuFFT on the GPU for %s.";
/// CUDA FFT error message.
TErrorMessage eRR_FMT_CUFFT_SETUP_FAILED
        = "Error: The cuFFT library failed to initialize for %s.";
/// CUDA FFT error message.
TErrorMessage ERR_FMT_CUFFT_INVALID_SIZE
        = "Error: cuFFT was given an invalid transform size for %s.";
/// CUDA FFT error message.
TErrorMessage ERR_FMT_CUFFT_UNALIGNED_DATA
        = "Error: Arrays for cuFFT was not properly aligned for %s.";
/// CUDA FFT error message.
TErrorMessage ERR_FMT_CUFFT_INCOMPLETE_PARAMETER_LIST
        = "Error: Missing parameters in the cuFFT call for %s.";
/// CUDA FFT error message.
TErrorMessage ERR_FMT_CUFFT_INVALID_DEVICE
        = "Error: cuFFT execution of the plan was performed on a different GPU than plan was "
          "created for %s.";
/// CUDA FFT error message.
TErrorMessage ERR_FMT_CUFFT_PARSE_ERROR
        = "Error: cuFFT internal plan database error for %s.";
/// CUDA FFT error message.
TErrorMessage ERR_FMT_CUFFT_NO_WORKSPACE
        = "Error: No workspace has been provided prior to cuFFT plan execution for %s.";
/// CUDA FFT error message.
TErrorMessage eRR_FMT_CUFFT_NOT_IMPLEMENTED
        = "Error: cuFFT feature is not implemented for %s.";
/// CUDA FFT error message.
TErrorMessage ERR_FMT_CUFFT_LICENSE_ERROR
        = "Error: cuFFT license error for %s.";
/// CUDA FFT error message.
TErrorMessage ERR_FMT_CUFFT_UNKNOWN_ERROR
        = "Error: cuFFT failed with unknown error for %s.";



//------------------------------------ CUDAParameters Class --------------------------------------//
/// CUDATuner error message
TErrorMessage ERR_FMT_BAD_DEVICE_IDX
        = "Error: Wrong CUDA device id %d. Allowed devices <0, %d>.";
/// CUDATuner error message
TErrorMessage ERR_FMT_NO_FREE_DEVICE
        = "Error: All CUDA-capable devices are busy or unavailable.";
/// CUDATuner error message
TErrorMessage ERR_FMT_DEVICE_IS_BUSY
        = "Error: CUDA device id %d is busy or unavailable.";

/// CUDAParameters error message
TErrorMessage ERR_FMT_INSUFFICIENT_CUDA_DRIVER
        = "Error: Insufficient CUDA driver version. The code needs CUDA version "
          "%d.%d but %d.%d is installed.";
/// CUDAParameters error message
TErrorMessage ERR_FM_CANNOT_READ_CUDA_VERSION
        = "Error: Insufficient CUDA driver version. Install the latest drivers.";
/// CUDAParameters error message
TErrorMessage ERR_FMT_GPU_NOT_SUPPORTED
        = "Error: CUDA device id %d is not supported by this k-Wave build.";


//------------------------------------- CheckErrors header ---------------------------------------//

        /// CUDAParameters error message
TErrorMessage ERR_FMT_GPU_ERROR
        = "GPU error: %s routine name: %s in file %s, line %d.";


#endif	/* ERROR_MESSAGES_H */
