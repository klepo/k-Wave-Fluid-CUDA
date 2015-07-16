/**
 * @file        TCUDATuner.h
 * @author      Jiri Jaros \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The implementation file for the class for setting CUDA kernel
 *              parameters.
 *
 * @version     kspaceFirstOrder3D 3.4
 * @date        04 November 2014, 14:40 (created) \n
 *              07 January  2015, 12:40 (revised)
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

#include <stdexcept>

#include <CUDA/CUDATuner.h>

#include <Utils/DimensionSizes.h>
#include <Utils/ErrorMessages.h>

using namespace std;

//----------------------------------------------------------------------------//
//--------------------------------- Macros -----------------------------------//
//----------------------------------------------------------------------------//

/**
 * Check errors of the CUDA routines and print error.
 * @param [in] code  - error code of last routine
 * @param [in] file  - The name of the file, where the error was raised
 * @param [in] line  - What is the line
 * @param [in] Abort - Shall the code abort?
 * @todo - check this routine and do it differently!
 */
inline void gpuAssert(cudaError_t code,
                      string file,
                      int line)
{
  if (code != cudaSuccess)
  {
    char ErrorMessage[256];
    sprintf(ErrorMessage,"GPUassert: %s %s %d\n",cudaGetErrorString(code),file.c_str(),line);

    // Throw exception
     throw std::runtime_error(ErrorMessage);
  }
}// end of gpuAssert
//------------------------------------------------------------------------------

/// Define to get the usage easier
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

//----------------------------------------------------------------------------//
//-------------------------------- Constants ---------------------------------//
//----------------------------------------------------------------------------//




//----------------------------------------------------------------------------//
//----------------------------- Initializations ------------------------------//
//----------------------------------------------------------------------------//
// set singleton instance flag
bool TCUDATuner::InstanceFlag = false;

// Set singleton flag
TCUDATuner* TCUDATuner::Instance = NULL;

// Set default size of the 3D block (use initialization list)
const dim3 TCUDATuner::DefaultNumberOfThreads3D {32, 8, 1};

//----------------------------------------------------------------------------//
//---------------------------------- Public ----------------------------------//
//----------------------------------------------------------------------------//

/**
 * Get instance of the singleton class
 * @return instance of the singleton class
 */
TCUDATuner* TCUDATuner::GetInstance()
{
  if(!InstanceFlag)
  {
    Instance = new TCUDATuner();
    InstanceFlag = true;
    return Instance;
  }
  else
  {
    return Instance;
  }
}// end of GetInstance
//------------------------------------------------------------------------------

/**
 * Default destructor for singleton class
 */
TCUDATuner::~TCUDATuner()
{
    delete Instance;
    Instance = NULL;
    InstanceFlag = false;
}// end of Destructor
//------------------------------------------------------------------------------

/**
 * Set size for 3D block
 * @param [in] X, Y, Z -  Sizes of the block
 */
void TCUDATuner::SetBlockSize3D(const int x, const int y, const int z)
{
  NumberOfThreads3D.x = x;
  NumberOfThreads3D.y = y;
  NumberOfThreads3D.z = z;
}// end of SetBlockSize3D
//------------------------------------------------------------------------------


/**
 * Selects GPU device for the code execution.
 * By default it chooses the first available device
 *
 * @param [in] DeviceIdx - this device will be used for execution.
 *                         If set to -1, let this routine to choose.
 *
 * @throw runtime_error will be thrown when it is not possible to use selected GPU
 */
void TCUDATuner::SetDevice(const int DeviceIdx)
{
  this->DeviceIdx = DeviceIdx;

  //choose the GPU device with the most global memory
  int NumOfDevices;
  gpuErrchk(cudaGetDeviceCount(&NumOfDevices));
  cudaGetLastError();

  //if the user does not provided a specific GPU, use the first one
  if (DeviceIdx == DefaultDeviceIdx)
  {
    bool DeviceFound = false;

    for (int testDevice = 0; testDevice < NumOfDevices; testDevice++)
    {
      // try to set and reset a device
      gpuErrchk(cudaSetDevice(testDevice));
      if (cudaDeviceReset() == cudaSuccess)
      { // success
        this->DeviceIdx = testDevice;
        DeviceFound = true;
        break;
      }
      // if the device was busy, clear errors
      cudaGetLastError();
    }

    if (!DeviceFound)
    {
      throw std::runtime_error(CUDATuner_ERR_FMT_NoFreeDevice);
    }
  }
  else // select a device the user wants
  {
    // check if the specified device is acceptable -
    // not busy, input parameter not out of bounds
    if ((this->DeviceIdx > NumOfDevices - 1) || (this->DeviceIdx < 0))
    {
      char ErrorMessage[256];
      sprintf(ErrorMessage, CUDATuner_ERR_FMT_WrongDeviceIdx, this->DeviceIdx, NumOfDevices-1);
      // Throw exception
      throw std::runtime_error(ErrorMessage);
     }

    // set the device and copy it's properties
    gpuErrchk(cudaSetDevice(this->DeviceIdx));
    if (cudaDeviceReset() != cudaSuccess)
    {
      char ErrorMessage[256];
      sprintf(ErrorMessage, CUDATuner_ERR_FMT_DeviceIsBusy, this->DeviceIdx);
      throw std::runtime_error(ErrorMessage);
    }
    cudaGetLastError();
  }

  // Read the device that was allocated
  gpuErrchk(cudaGetDevice(&this->DeviceIdx));
  gpuErrchk(cudaGetLastError());

  // Enable mapped memory
  gpuErrchk(cudaSetDeviceFlags(cudaDeviceMapHost));

    // Get Device name
  cudaDeviceProp DeviceProperties;
  gpuErrchk(cudaGetDeviceProperties(&DeviceProperties, this->DeviceIdx));
  DeviceName = DeviceProperties.name;

}// end of SetDevice
//------------------------------------------------------------------------------

/**
 * Test whether the device can handle the block sizes
 * @return  true if all is ok
 * @todo maybe better with exceptions
 */
bool TCUDATuner::CanDeviceHandleBlockSizes() const
{
  cudaDeviceProp DeviceProperties;
  gpuErrchk(cudaGetDeviceProperties(&DeviceProperties, DeviceIdx));

  // 1D block size check
  if(DeviceProperties.maxThreadsPerBlock < NumberOfThreads1D) return false;
  if(DeviceProperties.maxThreadsDim[0]   < NumberOfThreads1D) return false;


  // 3D block size check
  if(DeviceProperties.maxThreadsPerBlock < (NumberOfThreads3D.x * NumberOfThreads3D.y * NumberOfThreads3D.z))
  {
    return false;
  }
  if(DeviceProperties.maxThreadsDim[0] < NumberOfThreads3D.x) return false;
  if(DeviceProperties.maxThreadsDim[1] < NumberOfThreads3D.y) return false;
  if(DeviceProperties.maxThreadsDim[2] < NumberOfThreads3D.z) return false;

  // All is OK
  return true;
}// end of CanDeviceHandleBlockSizes
//------------------------------------------------------------------------------


/**
 * Get execution model for matrix size - generate grid sizes
 * @param [in] FullDimensinSizes    - Dimension sizes of the spatial space
 * @param [in] ReducedDimensinSizes - Dimension sizes of the k-space
 * @throw if it is possible to run the simulation.
 *
 */
void TCUDATuner::GenerateExecutionModelForMatrixSize(const TDimensionSizes & FullDimensionSizes,
                                                     const TDimensionSizes & ReducedDimensionSizes)
{
  // error checking
  if (FullDimensionSizes.GetElementCount() == 0)
  {
    throw runtime_error(CUDATuner_ERR_FMT_InvalidDimensions);
  }

  if (ReducedDimensionSizes.GetElementCount() == 0)
  {
    throw runtime_error(CUDATuner_ERR_FMT_InvalidDimensions);
  }

  //set the number of blocks for the 1D case
  size_t ElementCount = FullDimensionSizes.GetElementCount();

  // 1D Block
  NumberOfBlocks1D = 256;/*ElementCount / NumberOfThreads1D;
  if (ElementCount % NumberOfThreads1D > 0) NumberOfBlocks1D++; // add one block for the rest*/

  //3D block
  NumberOfBlocks3D.x = 2;//FullDimensionSizes.X / NumberOfThreads3D.x;
  NumberOfBlocks3D.y = 8;//FullDimensionSizes.Y / NumberOfThreads3D.y;
  NumberOfBlocks3D.z = 32;//FullDimensionSizes.Z / NumberOfThreads3D.z;

  NumberOfBlocks3DComplex.x = 2;
  NumberOfBlocks3DComplex.y = 8;
  NumberOfBlocks3DComplex.z = 32;



/*  NumberOfBlocks3D.x = FullDimensionSizes.X / NumberOfThreads3D.x;
  NumberOfBlocks3D.y = FullDimensionSizes.Y / NumberOfThreads3D.y;
  NumberOfBlocks3D.z = FullDimensionSizes.Z / NumberOfThreads3D.z;

  // 3D block checks for the rest
  if (FullDimensionSizes.X % NumberOfThreads3D.x > 0) NumberOfBlocks3D.x++;
  if (FullDimensionSizes.Y % NumberOfThreads3D.y > 0) NumberOfBlocks3D.y++;
  if (FullDimensionSizes.Z % NumberOfThreads3D.z > 0) NumberOfBlocks3D.z++;

  // 3D k-space block
  NumberOfBlocks3DComplex.x = ReducedDimensionSizes.X / NumberOfThreads3D.x;
  NumberOfBlocks3DComplex.y = ReducedDimensionSizes.Y / NumberOfThreads3D.y;
  NumberOfBlocks3DComplex.z = ReducedDimensionSizes.Z / NumberOfThreads3D.z;

  if (ReducedDimensionSizes.X % NumberOfThreads3D.x > 0) NumberOfBlocks3DComplex.x++;
  if (ReducedDimensionSizes.Y % NumberOfThreads3D.y > 0) NumberOfBlocks3DComplex.y++;
  if (ReducedDimensionSizes.Z % NumberOfThreads3D.z > 0) NumberOfBlocks3DComplex.z++;

  CheckAndAdjustGridSize3DToDeviceProperties();
*/

  //printf("Number of threads %d \n", GetNumberOfThreadsFor1D());
}// end of GenerateExecutionModelForMatrixSize
//------------------------------------------------------------------------------


/**
 * Get number of block for smaller matrices (usually for sampling purposes)
 * @todo Do I need this?
 * @param [in] NumberOfElements
 * @return number of blocks
 */
int TCUDATuner::GetNumberOfBlocksForSubmatrixWithSize(const size_t NumberOfElements) const
{
  size_t NumberOfBlocks;

  cudaDeviceProp DeviceProperties;
  gpuErrchk(cudaGetDeviceProperties(&DeviceProperties, DeviceIdx));

  NumberOfBlocks = NumberOfElements / NumberOfThreads1D;
  if (NumberOfElements % NumberOfThreads1D > 0) NumberOfBlocks++;

  if(NumberOfBlocks > DeviceProperties.maxGridSize[0]) NumberOfBlocks = DeviceProperties.maxGridSize[0];

  return static_cast<int> (NumberOfBlocks);
}// end of GetNumberOfBlocksForSubmatrixWithSize
//------------------------------------------------------------------------------

//----------------------------------------------------------------------------//
//-------------------------------- Protected ---------------------------------//
//----------------------------------------------------------------------------//



/**
 * Check the 3D grid size and adjust based on the device properties
 */
void TCUDATuner::CheckAndAdjustGridSize3DToDeviceProperties()
{
  cudaDeviceProp DeviceProperties;
  gpuErrchk(cudaGetDeviceProperties(&DeviceProperties, DeviceIdx));

  if(NumberOfBlocks1D > DeviceProperties.maxGridSize[0]) NumberOfBlocks1D = DeviceProperties.maxGridSize[0];


  if(NumberOfBlocks3D.x > DeviceProperties.maxGridSize[0]) NumberOfBlocks3D.x = DeviceProperties.maxGridSize[0];
  if(NumberOfBlocks3D.y > DeviceProperties.maxGridSize[1]) NumberOfBlocks3D.y = DeviceProperties.maxGridSize[1];
  if(NumberOfBlocks3D.z > DeviceProperties.maxGridSize[2]) NumberOfBlocks3D.z = DeviceProperties.maxGridSize[2];

  if(NumberOfBlocks3DComplex.x > DeviceProperties.maxGridSize[0]) NumberOfBlocks3DComplex.x = DeviceProperties.maxGridSize[0];
  if(NumberOfBlocks3DComplex.y > DeviceProperties.maxGridSize[1]) NumberOfBlocks3DComplex.y = DeviceProperties.maxGridSize[1];
  if(NumberOfBlocks3DComplex.z > DeviceProperties.maxGridSize[2]) NumberOfBlocks3DComplex.z = DeviceProperties.maxGridSize[2];

}// end of CheckAndAdjustGridSize3DToDeviceProperties
//------------------------------------------------------------------------------



//----------------------------------------------------------------------------//
//--------------------------------- Private ----------------------------------//
//----------------------------------------------------------------------------//

/**
 * Default constructor.
 */
TCUDATuner::TCUDATuner() :
        DeviceIdx(DefaultDeviceIdx),
        NumberOfThreads1D(DefaultNumberOfThreads1D), NumberOfBlocks1D(0),
        NumberOfThreads3D(DefaultNumberOfThreads3D), NumberOfBlocks3D(0,0,0),
        NumberOfBlocks3DComplex(0,0,0),
        DeviceName()
{
//printf("Number of threads %d \n", GetNumberOfThreadsFor1D());
}// end of TCUDATuner()
//------------------------------------------------------------------------------
