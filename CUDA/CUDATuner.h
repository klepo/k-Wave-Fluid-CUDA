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
