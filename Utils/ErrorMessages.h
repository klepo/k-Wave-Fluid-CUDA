/**
 * @file        ErrorMessages.h
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file containing all error messages of the project.
 *
 * @version     kspaceFirstOrder3D 3.4
 * @date        09 August   2011, 12:34 (created) \n
 *              17 December 2014, 13:18 (revised)
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


#ifndef ERRORMESSAGES_H
#define	ERRORMESSAGES_H

//----------------------------- HDF5 error messages --------------------------//
/**
 * @typedef TErrorMessage
 * @brief Datatype for error messages
 */
typedef const char * const TErrorMessage;

/// HDF5 error message
TErrorMessage HDF5_ERR_FMT_FileNotCreated          = "Error: File \"%s\" could not be created!\n";
/// HDF5 error message
TErrorMessage HDF5_ERR_FMT_FileCannotRecreated     = "Error: Cannot recreate an opened file \"%s\"!\n";
/// HDF5 error message
TErrorMessage HDF5_ERR_FMT_FileCannotReopen        = "Error: Cannot reopen an opened file \"%s\"!\n";
/// HDF5 error message
TErrorMessage HDF5_ERR_FMT_FileNotClosed           = "Error: File \"%s\" could not be closed!\n";
/// HDF5 error message
TErrorMessage HDF5_ERR_FMT_CouldNotWriteTo         = "Error: Could not write into \"%s\" dataset!\n";
/// HDF5 error message
TErrorMessage HDF5_ERR_FMT_CouldNotReadFrom        = "Error: Could not read from \"%s\" dataset!\n";
/// HDF5 error message
TErrorMessage HDF5_ERR_FMT_WrongDimensionSizes     = "Error: Dataset \"%s\"  has wrong dimension sizes!\n";
/// HDF5 error message
TErrorMessage HDF5_ERR_FMT_FileNotOpened           = "Error: File \"%s\" could not be opened!\n";
/// HDF5 error message
TErrorMessage HDF5_ERR_FMT_NotHDF5File             = "Error: File \"%s\" is not a valid HDF5 file!\n";
/// HDF5 error message
TErrorMessage HDF5_ERR_FMT_DatasetNotOpened        = "Error: File \"%s\" could not open dataset \"%s\"!\n";
/// HDF5 error message
TErrorMessage HDF5_ERR_FMT_CouldNotSetCompression  = "Error: File \"%s\", dataset \"%s\" could set compression level [%ld]!\n";
/// HDF5 error message
TErrorMessage HDF5_ERR_FMT_BadAttributeValue       = "Error: Bad attribute value: [%s,%s] = %s";
/// HDF5 error message
TErrorMessage HDF5_ERR_FMT_CouldNotWriteToAttribute  = "Error: Could not write into \"%s\" attribute of \"%s\" dataset!\n";
/// HDF5 error message
TErrorMessage HDF5_ERR_FMT_CouldNotReadFromAttribute = "Error: Could not read from \"%s\" attribute of \"%s\" dataset!\n";
/// HDF5 error message
TErrorMessage HDF5_ERR_FMT_GroupNotCreated          = "Error: Could not create group \"%s\" in file \"%s\"!\n";
/// HDF5 error message
TErrorMessage HDF5_ERR_FMT_GroupNotOpened           = "Error: Could not open group \"%s\" in file \"%s\"!\n";
/// HDF5 error message
TErrorMessage HDF5_ERR_FMT_BadInputFileType         = "Error: The input file has not a valid format!\n";
/// HDF5 error message
TErrorMessage HDF5_ERR_FMT_BadOutputFileType        = "Error: The output file has not a valid format!\n";
/// HDF5 error message
TErrorMessage HDF5_ERR_FMT_BadCheckpointFileType    = "Error: The checkpoint file has not a valid format!\n";

//---------------------------------- Matrix Classes  -------------------------//

/// Matrix class error message
TErrorMessage  Matrix_ERR_FMT_Not_Enough_Memory  = "Error: Class %s: Memory allocation failed: Not Enough Memory\n";
/// Matrix class error message
TErrorMessage  Matrix_ERR_FMT_MatrixNotFloat     = "Error: Matrix [%s] data type is not of single precision floating point!\n";
/// Matrix class error message
TErrorMessage  Matrix_ERR_FMT_MatrixNotReal      = "Error: Matrix [%s] domain is not real!\n";
/// Matrix class error message
TErrorMessage  Matrix_ERR_FMT_MatrixNotComplex   = "Error: Matrix [%s] domain is not complex!\n";
/// Matrix class error message
TErrorMessage  Matrix_ERR_FMT_MatrixNotLong      = "Error: Matrix [%s] data type is not unsigned long!\n";

//--------------------------------- Matrix Container  ------------------------//

/// Matrix container error message
TErrorMessage  MatrixContainer_ERR_FMT_RecordUnknownDistributionType =
"K-Space panic: Matrix [%s] has unknown distribution type in the C++ code!\n\
        [File, line] : [%s,%d]!\n";

/// Matrix container error message
TErrorMessage  MatrixContainer_ERR_FMT_ReloactaionError =
"K-Space panic: Matrix [%s] is being reallocated!\n\
        [File, line] : [%s,%d]!\n";


//-------------------------- Command line Parameters  ------------------------//

/// Command line parameters error message
TErrorMessage CommandlineParameters_ERR_FMT_NoVerboseIntreval        = "Command line parsing error: No or invalid verbose interval provided!\n";
/// Command line parameters error message
TErrorMessage CommandlineParameters_ERR_FMT_NoThreadNumbers          = "Command line parsing error: No or invalid number of CPU threads!\n";
/// Command line parameters error message
TErrorMessage CommandlineParameters_ERR_FMT_NoGPUNumbers             = "Command line parsing error: No or invalid number for GPU device!\n";
/// Command line parameters error message
TErrorMessage CommandlineParameters_ERR_FMT_NoCompressionLevel       = "Command line parsing error: No or invalid compression level!\n";
/// Command line parameters error message
TErrorMessage CommandlineParameters_ERR_FMT_NoStartTimestep          = "Command line parsing error: No or invalid collection start time step!\n";
/// Command line parameters error message
TErrorMessage CommandlineParameters_ERR_FMT_NoBenchmarkTimeStepCount = "Command line parsing error: No or invalid benchmark time step count!\n";
/// Command line parameters error message
TErrorMessage CommandlineParameters_ERR_FMT_NoInputFile =  "Error: The input file was not specified!\n";
/// Command line parameters error message
TErrorMessage CommandlineParameters_ERR_FMT_NoOutputFile=  "Error: The output file was not specified!\n";
/// Command line parameters error message
TErrorMessage CommandlineParameters_ERR_FMT_NoCheckpointFile         = "Error: The checkpoint file was not specified!\n";
/// Command line parameters error message
TErrorMessage CommandlineParameters_ERR_FMT_NoCheckpointInterval     = "Error: The checkpoint interval was not specified!\n";

/// Command line parameters error message
TErrorMessage Parameters_ERR_FMT_Illegal_alpha_power_value = "Error: Illegal value of alpha_power!";
/// Command line parameters error message
TErrorMessage Parameters_ERR_FMT_Illegal_StartTime_value   = "Error: The start index is out of the simulation span <%ld, %ld>!\n";
/// Command line parameters error message
TErrorMessage Parameters_ERR_FMT_IncorrectInputFileFormat = "Error: Incorrect input file\"%s\" format!\n";
/// Command line parameters error message
TErrorMessage Parameters_ERR_FMT_IncorrectMajorHDF5FileVersion = "Error: Incorrect major version of the HDF5 file %s (expected is %s)!\n";
/// Command line parameters error message
TErrorMessage Parameters_ERR_FMT_IncorrectMinorHDF5FileVersion = "Error: Incorrect minor version of the HDF5 file %s (expected is %s)!\n";
/// Command line parameters error message
TErrorMessage Parameters_ERR_FMT_WrongSensorMaskType = "Error: The sensor mask type specified in the input file is not supported! \n";
/// Command line parameters error message
TErrorMessage Parameters_ERR_FMT_UNonStaggeredNotSupportedForFile10 = "Error: --u_non_staggered_raw is not supported along with the input file of the version 1.0! \n";

//------------------------------- CUDA  --------------------------------------//

/// CUDA error message.
TErrorMessage CUDA_ERR_FMT_BadBlocksSize = "Error: The device %d cannot handle either the \n \
                                                 \t 1D blocksize of %d\n \t or \n \t 3D blocksize of %d,%d,%d \n";


//-------------------------------- CUDA FFT Errors ---------------------------//
/// CUDA FFT error message.
TErrorMessage CUFFTComplexMatrix_ERR_FMT_INVALID_PLAN         = "Error: cuFFT was passed an invalid plan handle for %s! \n";
/// CUDA FFT error message.
TErrorMessage CUFFTComplexMatrix_ERR_FMT_CUFFT_ALLOC_FAILED   = "Error: cuFFT failed to allocate GPU or CPU memory for %s! \n";
/// CUDA FFT error message.
TErrorMessage CUFFTComplexMatrix_ERR_FMT_CUFFT_INVALID_TYPE   = "Error: cuFFT invalid type for of the transform for %s! \n";
/// CUDA FFT error message.
TErrorMessage CUFFTComplexMatrix_ERR_FMT_CUFFT_INVALID_VALUE  = "Error: cuFFT was given an invalid pointer or parameter for %s! \n";
/// CUDA FFT error message.
TErrorMessage CUFFTComplexMatrix_ERR_FMT_CUFFT_INTERNAL_ERROR = "Error: Driver or internal cuFFT library error for %s! \n";
/// CUDA FFT error message.
TErrorMessage CUFFTComplexMatrix_ERR_FMT_CUFFT_EXEC_FAILED    = "Error: Failed to execute an ccFFT on the GPU for %s! \n";
/// CUDA FFT error message.
TErrorMessage CUFFTComplexMatrix_ERR_FMT_CUFFT_SETUP_FAILED   = "Error: The cuFFT library failed to initialize for %s! \n";
/// CUDA FFT error message.
TErrorMessage CUFFTComplexMatrix_ERR_FMT_CUFFT_INVALID_SIZE   = "Error: cuFFT was given an invalid transform size for %s! \n";
/// CUDA FFT error message.
TErrorMessage CUFFTComplexMatrix_ERR_FMT_CUFFT_UNALIGNED_DATA = "Error: Arrays for cuFFT was not properly aligned for %s! \n";
/// CUDA FFT error message.
TErrorMessage CUFFTComplexMatrix_ERR_FMT_CUFFT_INCOMPLETE_PARAMETER_LIST = "Error: Missing parameters in the cuFFT call for %s! \n";
/// CUDA FFT error message.
TErrorMessage CUFFTComplexMatrix_ERR_FMT_CUFFT_INVALID_DEVICE  = "Error: cuFFT execution of a plan was on different GPU than plan creation for %s! \n";
/// CUDA FFT error message.
TErrorMessage CUFFTComplexMatrix_ERR_FMT_CUFFT_PARSE_ERROR     = "Error: cuFFT internal plan database error for %s! \n";
/// CUDA FFT error message.
TErrorMessage CUFFTComplexMatrix_ERR_FMT_CUFFT_NO_WORKSPACE    = "Error: No workspace has been provided prior to cuFFT plan execution for %s! \n";
/// CUDA FFT error message.
TErrorMessage CUFFTComplexMatrix_ERR_FMT_CUFFT_NOT_IMPLEMENTED = "Error: cuFFT feature is not implemented for %s! \n";
/// CUDA FFT error message.
TErrorMessage CUFFTComplexMatrix_ERR_FMT_CUFFT_LICENSE_ERROR   = "Error: cuFFT license error for %s! \n";
/// CUDA FFT error message.
TErrorMessage CUFFTComplexMatrix_ERR_FMT_CUFFT_UNKNOWN_ERROR   = "Error: cuFFT failed with unknown error for %s! \n";


//------------------------- KSpaceFirstOrder3DSolver Classes  ----------------//

/// KSpaceFirstOrder3DSolver error message
TErrorMessage KSpaceFirstOrder3DSolver_ERR_FMT_IncorrectCheckpointFileFormat
  = "Error: Incorrect checkpoint file \"%s\" format!\n";

/// KSpaceFirstOrder3DSolver error message
TErrorMessage KSpaceFirstOrder3DSolver_ERR_FMT_IncorrectOutputFileFormat
  = "Error: Incorrect output file \"%s\" format!\n";

/// KSpaceFirstOrder3DSolver error message
TErrorMessage KSpaceFirstOrder3DSolver_ERR_FMT_CheckpointDimensionsDoNotMatch
  = "Error: The dimensions [%ld, %ld, %ld] of the checkpoint file don't match the simulation dimensions [%ld, %ld, %ld]! \n";

/// KSpaceFirstOrder3DSolver error message
TErrorMessage KSpaceFirstOrder3DSolver_ERR_FMT_OutputDimensionsDoNotMatch
  = "Error: The dimensions [%ld, %ld, %ld] of the output file don't match the simulation dimensions [%ld, %ld, %ld]!\n";

//----------------------------------- CUDATuner Class  -----------------------//

/// CUDATuner error message
TErrorMessage CUDATuner_ERR_FMT_WrongDeviceIdx    = "Error: Wrong CUDA device idx %d. Allowed devices 0-%d!\n";
/// CUDATuner error message
TErrorMessage CUDATuner_ERR_FMT_NoFreeDevice      = "Error: All CUDA-capable devices are busy or unavailable!\n";
/// CUDATuner error message
TErrorMessage CUDATuner_ERR_FMT_DeviceIsBusy      = "Error: CUDA device idx %d is busy or unavailable! \n";
/// CUDATuner error message
TErrorMessage CUDATuner_ERR_FMT_InvalidDimensions = "Error: Invalid simulation dimensions (zero when generating the execution model)!\n";

#endif	/* ERRORMESSAGES_H */




