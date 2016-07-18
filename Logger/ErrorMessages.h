/**
 * @file        ErrorMessages.h
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file containing routines for error messages.
 *              The actual error messages are included form different headers
 *              based on the operating system (encoding)
 *
 * @version     kspaceFirstOrder3D 3.4
 * @date        09 August   2011, 12:34 (created) \n
 *              13 July     2016, 11:12 (revised)
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

/**
 * Checks CUDA errors, create an error message and throw an exception.
 * The template parameter should be set to true for the whole code when debugging
 * kernel related errors.
 * Setting it to true for production run will cause IO sampling and storing not
 * to be overlapped
 *
 * @param [in] error_cod    - error produced by a cuda routine
 * @param [in] routine_name - function where the error happened
 * @param [in] file_name    - file where the error happened
 * @param [in] line_number   - line where the error happened
 */
template <bool ForceSynchronisation = false>
inline void CheckErrors(const cudaError_t error_code,
                        const char*       routine_name,
                        const char*       file_name,
                        const int         line_number)
{
  if (ForceSynchronisation)
  {
    cudaDeviceSynchronize();
  }

  if (error_code != cudaSuccess)
  {
    char ErrorMessage[512];
    snprintf(ErrorMessage,
             512,
             CUDACheckErrors_ERR_FM_GPU_Error,
             cudaGetErrorString(error_code), routine_name,file_name, line_number);

    // Throw exception
     throw std::runtime_error(ErrorMessage);
  }
}// end of cudaCheckErrors
//------------------------------------------------------------------------------

/**
 * @brief Macro checking cuda errors and printing the file name and line
 */
#define checkCudaErrors(val) CheckErrors ( (val), #val, __FILE__, __LINE__ )


#endif	/* ERROR_MESSAGES_H */




