/**
 * @file        CUDADeviceConstants.cu
 * @author      Jiri Jaros \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The implementation file for the class for storing constants residing in CUDA constant memory.
 *
 * @version     kspaceFirstOrder3D 3.4
 * @date        17 February 2016, 10:53 (created) \n
 *              12 April    2016, 15:02 (revised)
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

#include <string>
#include <stdexcept>

#include <Parameters/CUDADeviceConstants.cuh>
#include <Utils/ErrorMessages.h>


//----------------------------------------------------------------------------//
//-------------------------------- Constants ---------------------------------//
//----------------------------------------------------------------------------//


//----------------------------------------------------------------------------//
//-------------------------------- Variables ---------------------------------//
//----------------------------------------------------------------------------//



/**
 * @variable CUDADeviceConstants
 * @brief    This variable holds basic simulation constants for GPU.
 * @details  This variable holds necessary simulation constants in the CUDA GPU.
 *           memory. This variable is imported as extern into other CUDA units
 */
__constant__ TCUDADeviceConstants CUDADeviceConstants;


//----------------------------------------------------------------------------//
//----------------------------- Global routines ------------------------------//
//----------------------------------------------------------------------------//


//----------------------------------------------------------------------------//
//---------------------------------- Public ----------------------------------//
//----------------------------------------------------------------------------//

__host__ void TCUDADeviceConstants::SetUpCUDADeviceConstatns()
{
  checkCudaErrors(cudaMemcpyToSymbol(CUDADeviceConstants, this, sizeof(TCUDADeviceConstants)));
}// end of SetUpCUDADeviceConstatns
//------------------------------------------------------------------------------