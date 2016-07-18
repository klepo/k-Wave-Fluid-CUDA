/**
 * @file        CUDAParameters.h
 * @author      Jiri Jaros \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file for the class for setting CUDA kernel parameters.
 *
 * @version     kspaceFirstOrder3D 3.4
 * @date        12 November 2015, 16:49 (created) \n
 *              14 March    2016, 14:47 (revised)
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

#ifndef CUDA_PARAMETERS_H
#define CUDA_PARAMETERS_H

#include <string>
#include <iostream>
#include <cuda_runtime.h>

#include <Utils/DimensionSizes.h>


// forward declaration
//class TParameters;

/**
 * @class   TCUDAParameters
 * @brief   Class responsible for CUDA runtime setup
 * @details Class responsible for selecting a CUDA device, block and grid dimensions,
 *          etc. \n
 *          The class can only by constructed from inside TPrameters and there
 *          mustn't be more than 1 instance in the code
 *
 */
class TCUDAParameters
{
  public:
    friend class TParameters;
    /// Destructor, no need to be virtual
    ~TCUDAParameters() {}

    /// Get Idx of the device being used
    int GetDeviceIdx()                 const { return DeviceIdx;               }
    /// Get number of threads for 1D block used by kSpaceSolver.
    int GetSolverBlockSize1D()         const { return SolverBlockSize1D;       }
    /// Get number of block for 1D grid used by kSpaceSolver.
    int GetSolverGridSize1D()          const { return SolverGridSize1D;        }


    /// Get block size for the transposition kernels
    dim3 GetSolverTransposeBlockSize() const { return SolverTransposeBlockSize;}
    /// Get grid size for the transposition kernels
    dim3 GetSolverTransposeGirdSize()  const { return SolverTransposeGirdSize; }


    /// Get number of threads for the 1D data sampling kernels
    int GetSamplerBlockSize1D()        const { return SamplerBlockSize1D;      }
    /// Get Number of blocks for the 1D data sampling kernels
    int GetSamplerGridSize1D()         const { return SamplerGridSize1D;       }

    /// Get the name of the device used.
    std::string GetDeviceName()        const;

    /// Select cuda device for execution.
    void SelectDevice(const int DeviceIdx = DefaultDeviceIdx);
    /// Set kernel configurations based on the simulation parameters
    void SetKernelConfiguration();

    /// Upload useful simulation constants into device constant memory
    void SetUpDeviceConstants();

    /// Return properties of currently used GPU
    cudaDeviceProp GetDeviceProperties() const {return DeviceProperties;};

  protected:
    /// Default constructor
    TCUDAParameters();

    /// Copy constructor not allowed for public.
    TCUDAParameters(const TCUDAParameters& src);

    /// operator = not allowed for public.
    TCUDAParameters& operator = (const TCUDAParameters& src);

    /// Check whether the CUDA driver version installed is sufficient for the code
    void CheckCUDAVersion();

    /// Check whether the code was compiled for a given SM model
    bool CheckCUDACodeVersion();


  private:

    /// Index of the device the code is being run on.
    int  DeviceIdx;

    /// Number of threads for 1D block used by kSpaceSolver.
    int  SolverBlockSize1D;
    /// Number of block for 1D grid used by kSpaceSolver.
    int  SolverGridSize1D;

    /// Block size for the transposition kernels
    dim3 SolverTransposeBlockSize;
    /// Grid size for the transposition kernels
    dim3 SolverTransposeGirdSize;


    /// Number of threads for the 1D data sampling kernels
    int SamplerBlockSize1D;
    /// Number of blocks for the 1D data sampling kernels
    int SamplerGridSize1D;

    /// Device properties of the selected GPU.
    cudaDeviceProp DeviceProperties;

    /// Undefined block or grid size
    static const int UndefinedSize = -1;

    /// Default Device Index - no default GPU
    static const int DefaultDeviceIdx = -1;
};

#endif /* CUDAPARAMETERS_H */

