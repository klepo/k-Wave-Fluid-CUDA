/**
 * @file        CUDATuner.h
 * @author      Jiri Jaros \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file for the class for setting CUDA kernel parameters.
 *
 * @version     kspaceFirstOrder3D 3.4
 * @date        04 November 2014, 14:40 (created) \n
 *              17 December 2014, 20:36 (revised)
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

#ifndef CUDA_TUNER_H
#define CUDA_TUNER_H

#include <string>
#include <iostream>
#include <cuda_runtime.h>

#include <Utils/DimensionSizes.h>

/**
 * @class CUDATnuer
 * @brief Class responsible cuda blocks and grid seizes
 * @details Class responsible cuda blocks and grid seizes and device selection
 *
 * @todo - needs a bit of investigation -> Move to TParameters?
 *         There is a collision with TCommandline parameters
 */
class TCUDATuner
{
  public:
    /// Get instance of the singleton class.
    static TCUDATuner* GetInstance();

    ///Destructor.
    virtual ~TCUDATuner();

    /// Set block size of 1D Block.
    void SetBlockSize1D(const int BlockSize1D) { NumberOfThreads1D = BlockSize1D;};
    /// Set size for 3D block.
    void SetBlockSize3D(const int x, const int y, const int z);
    /// Set size for 3D block.
    void SetBlockSize3D(const dim3 & BlockSize3D) { NumberOfThreads3D = BlockSize3D;};

    /// Set CUDA device for execution (-1 - let system to select the best on).
    void SetDevice(const int DeviceIdx = DefaultDeviceIdx);
    /// Get CUDA Device.
    int  GetDevice() const {return DeviceIdx;};

    /// Get CUDA device name.
    std::string GetDeviceName() const {return DeviceName;};

    /// Can the device handle this block size?
    bool CanDeviceHandleBlockSizes() const;
    /// Generate grid sizes for matrix
    void GenerateExecutionModelForMatrixSize(const TDimensionSizes & FullDimensionSizes,
                                             const TDimensionSizes & ReducedDimensionSizes);

    /// Get number of block for smaller 1D matrix
    int GetNumberOfBlocksForSubmatrixWithSize(const size_t NumberOfElements) const;

    /// Get number of threads for 1D block.
    int GetNumberOfThreadsFor1D() const {return NumberOfThreads1D;};
    /// Get number of block for 1D grid.
    int GetNumberOfBlocksFor1D()  const {return NumberOfBlocks1D;};

    /// Get number of threads for 3D block.
    dim3 GetNumberOfThreadsFor3D() const {return NumberOfThreads3D;};
    /// Get number for blocks for 3D grid.
    dim3 GetNumberOfBlocksFor3D()  const {return NumberOfBlocks3D;};
    /// Get number of block for 3D grid running in k-space (over comples matrices).
    dim3 GetNumberOfBlocksFor3DComplex() const {return NumberOfBlocks3DComplex;};

    /// Show GPU parameters.
    void ShowGPUParams();
    /// Set GPU parameters.
    void SetGPUParams();
    /// Set Default GPU parameters
    void SetDefaultGPUParams();

  protected:
    /// Device index - where to run the code.
    int  DeviceIdx;

    /// Number of threads for 1D block.
    int  NumberOfThreads1D;
    /// Number of block for 1D grid.
    int  NumberOfBlocks1D;

    /// Number of Block in 3D grid.
    dim3 NumberOfThreads3D;
    /// Number of block for 3D grid.
    dim3 NumberOfBlocks3D;

   /// Number of Blocks in 3D complex grid
    dim3 NumberOfBlocks3DComplex;

    /// Type of the device used.
    std::string DeviceName;

    // Check the grid size and if it's bigger than supported adjust it.
    void CheckAndAdjustGridSize3DToDeviceProperties();

  private:
    /// Default constructor for singleton.
    TCUDATuner();
    /// Singleton flag.
    static bool InstanceFlag;
    ///Singleton instance.
    static TCUDATuner * Instance;

    /// Default device -1 - let the class to choose
    static const int  DefaultDeviceIdx   = -1;
    /// Default size of 1D block (empirically tested of Fermi and Kepler)
    static const int  DefaultNumberOfThreads1D = 128;
    /// Default size of 3D block (empirically tested of Fermi and Kepler)
    static const dim3 DefaultNumberOfThreads3D; //initialized in the cpp file= {128, 1, 1};

};
#endif /* CUDA_TUNER_H */
