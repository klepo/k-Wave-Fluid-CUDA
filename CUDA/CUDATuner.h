/**
 * @file        CUDATuner.h
 * @author      Jiri Jaros \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file for the class for setting CUDA kernel parameters.
 *
 * @version     kspaceFirstOrder3D 3.3
 * @date        04 November 2014, 14:40 (created) \n
 *              04 November 2014, 14:47 (revised)
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

#ifndef CUDATUNER_H
#define CUDATUNER_H

#include <string>
#include <iostream>
#include <cuda_runtime.h>

class CUDATuner
{
    private:
        static bool CUDATunerInstanceFlag;
        static CUDATuner* CUDATunerSingle;
        CUDATuner();

    public:
        static CUDATuner* GetInstance();
        virtual ~CUDATuner();

        void Set1DBlockSize(int i);
        void Set3DBlockSize(int x, int y, int z);
        bool SetDevice(int);
        int  GetDevice();
        std::string GetDeviceName();
        bool CanDeviceHandleBlockSizes();
        bool GenerateExecutionModelForMatrixSize(size_t X,
                                                 size_t Y,
                                                 size_t Z);
        int GetNumberOfBlocksForSubmatrixWithSize(size_t number_of_elements);

        //accessors
        int GetNumberOfThreadsFor1D()
        {
            return default_number_of_threads_1D;
        }

        int GetNumberOfBlocksFor1D()
        {
            return default_number_of_blocks_1D;
        }

        dim3 GetNumberOfThreadsFor3D()
        {
            return default_number_of_threads_3D;
        }

        dim3 GetNumberOfBlocksFor3D()
        {
            return default_number_of_blocks_3D;
        }

        dim3 GetNumberOfBlocksFor3DComplex()
        {
            return default_number_of_blocks_3D_complex;
        }

        void ShowGPUParams();
        void SetGPUParams();
        void SetDefaultGPUParams();

        protected:
        //variables
        int  device_id;

        int  default_number_of_blocks_1D;
        int  default_number_of_threads_1D;

        dim3 default_number_of_blocks_3D;
        dim3 default_number_of_threads_3D;

        dim3 default_number_of_blocks_3D_complex;

        std::string device_type;

        //functions
        void IsTheNumberOfBlocksGreaterThanSupportedOnDevice();

};
#endif /* CUDATUNER_H */
