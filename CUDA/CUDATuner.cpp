
#include "CUDATuner.h"

using namespace std;

//singleton variables
bool CUDATuner::CUDATunerInstanceFlag = false;
CUDATuner* CUDATuner::CUDATunerSingle = NULL;

//singleton functions
CUDATuner* CUDATuner::GetInstance()
{

    if(!CUDATunerInstanceFlag)
    {
        CUDATunerSingle = new CUDATuner();
        CUDATunerInstanceFlag = true;
        return CUDATunerSingle;
    }
    else
    {
        return CUDATunerSingle;
    }

}

CUDATuner::CUDATuner()
{
    default_number_of_threads_1D = 128;
    default_number_of_threads_3D.x = 128;
    default_number_of_threads_3D.y = 1;
    default_number_of_threads_3D.z = 1;
}

CUDATuner::~CUDATuner()
{
    delete CUDATunerSingle;
    CUDATunerSingle = NULL;
    CUDATunerInstanceFlag = false;
}

void CUDATuner::Set1DBlockSize(int i)
{
    if(i != 0){
        default_number_of_threads_1D = i;
    }
}

void CUDATuner::Set3DBlockSize(int x, int y, int z)
{
    if(x != 0 || y != 0 || z != 0){
        default_number_of_threads_3D.x = x;
        default_number_of_threads_3D.y = y;
        default_number_of_threads_3D.z = z;
    }
}

bool CUDATuner::SetDevice(int id)
{
    device_id = id;

    //set a cuda complient gpu device if non was set by the user
    //(as a command line argument)
    if(device_id == -1){
        //choose the GPU device with the most global memory
        int num_devices, device;
        cudaGetDeviceCount(&num_devices);
        if (num_devices > 1) {
            size_t max_global_memory = 0, best_device = 0;
            for (device = 0; device < num_devices; device++) {
                cudaDeviceProp properties;
                cudaGetDeviceProperties(&properties, device);
                if (max_global_memory < properties.totalGlobalMem) {
                    max_global_memory = properties.totalGlobalMem;
                    best_device = device;
                }
            }
            device_id = best_device;
        } else {
            //there is only 1 gpu so use it
            device_id = 0;
        }
    }

    // check if the specified device is acceptable
    int DeviceCount;
    cudaGetDeviceCount(&DeviceCount);
    if (static_cast<int>(device_id) >= static_cast<int>(DeviceCount)){
        cerr << "ERROR: Wrong device id : " << device_id <<" Allowed 0-" <<
            DeviceCount-1 << endl;
        return false;
    }

    // set the device and copy it's properties
    cudaSetDevice(device_id);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    device_type = prop.name;

    return true;
}

int  CUDATuner::GetDevice()
{
    return device_id;
}

string CUDATuner::GetDeviceName()
{
    return device_type;
}

bool CUDATuner::CanDeviceHandleBlockSizes()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);

    //1d block size checks
    if(prop.maxThreadsPerBlock < default_number_of_threads_1D){
        return false;
    }
    if(prop.maxThreadsDim[0] < default_number_of_threads_1D){
        return false;
    }

    //3d block size checks
    if(prop.maxThreadsPerBlock < (default_number_of_threads_3D.x*
                                  default_number_of_threads_3D.y*
                                  default_number_of_threads_3D.z)){
        return false;
    }
    if(prop.maxThreadsDim[0] < default_number_of_threads_3D.x){
        return false;
    }
    if(prop.maxThreadsDim[1] < default_number_of_threads_3D.y){
        return false;
    }
    if(prop.maxThreadsDim[2] < default_number_of_threads_3D.z){
        return false;
    }
    return true;
}

bool CUDATuner::GenerateExecutionModelForMatrixSize(
        size_t full_dimension_sizes_X,
        size_t full_dimension_sizes_Y,
        size_t full_dimension_sizes_Z)
{
    if(full_dimension_sizes_X == 0 || full_dimension_sizes_Y == 0 ||
            full_dimension_sizes_Z == 0){
        cerr << "ERROR: A dimension was zero when generating the"
             << " execution model." << endl;
        return false;
    }

    //set the number of blocks for the 1D case
    size_t total_size =
        full_dimension_sizes_X*full_dimension_sizes_Y*full_dimension_sizes_Z;

    if(total_size % default_number_of_threads_1D == 0){
        default_number_of_blocks_1D =
            total_size / default_number_of_threads_1D;
    } else{
        default_number_of_blocks_1D =
            total_size / default_number_of_threads_1D + 1;
    }

    //set the number of blocks for the 3D X case
    if(full_dimension_sizes_X % default_number_of_threads_3D.x == 0){
        default_number_of_blocks_3D.x =
            full_dimension_sizes_X / default_number_of_threads_3D.x;
    } else{
        default_number_of_blocks_3D.x =
            full_dimension_sizes_X / default_number_of_threads_3D.x + 1;
    }

    //set the number of blocks for the 3D Y case
    if(full_dimension_sizes_Y % default_number_of_threads_3D.y == 0){
        default_number_of_blocks_3D.y =
            full_dimension_sizes_Y / default_number_of_threads_3D.y;
    } else{
        default_number_of_blocks_3D.y =
            full_dimension_sizes_Y / default_number_of_threads_3D.y + 1;
    }

    //set the number of blocks for the 3D Z case
    if(full_dimension_sizes_Z % default_number_of_threads_3D.z == 0){
        default_number_of_blocks_3D.z =
            full_dimension_sizes_Z / default_number_of_threads_3D.z;
    } else{
        default_number_of_blocks_3D.z =
            full_dimension_sizes_Z / default_number_of_threads_3D.z + 1;
    }

    //set the number of blocks for the 3D complex case
    size_t reduced_dimension_sizes_X = (full_dimension_sizes_X/2)+1;
    if(reduced_dimension_sizes_X % default_number_of_threads_3D.x == 0){
        default_number_of_blocks_3D_complex.x =
            reduced_dimension_sizes_X / default_number_of_threads_3D.x;
    } else{
        default_number_of_blocks_3D_complex.x =
            reduced_dimension_sizes_X / default_number_of_threads_3D.x + 1;
    }
    default_number_of_blocks_3D_complex.y = default_number_of_blocks_3D.y;
    default_number_of_blocks_3D_complex.z = default_number_of_blocks_3D.z;

    IsTheNumberOfBlocksGreaterThanSupportedOnDevice();

    return true;
}

int CUDATuner::GetNumberOfBlocksForSubmatrixWithSize(size_t number_of_elements)
{
    if(number_of_elements % default_number_of_threads_1D == 0){
            return number_of_elements / default_number_of_threads_1D;
    } else{
            return number_of_elements / default_number_of_threads_1D + 1;
    }
}

void CUDATuner::IsTheNumberOfBlocksGreaterThanSupportedOnDevice()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);

    if(default_number_of_blocks_1D > prop.maxGridSize[0]){
        default_number_of_blocks_1D = prop.maxGridSize[0];
    }

    if(default_number_of_blocks_3D.x > prop.maxGridSize[0]){
        default_number_of_blocks_3D.x = prop.maxGridSize[0];
    }
    if(default_number_of_blocks_3D.y > prop.maxGridSize[1]){
        default_number_of_blocks_3D.y = prop.maxGridSize[1];
    }
    if(default_number_of_blocks_3D.z > prop.maxGridSize[2]){
        default_number_of_blocks_3D.z = prop.maxGridSize[2];
    }

    if(default_number_of_blocks_3D_complex.x > prop.maxGridSize[0]){
        default_number_of_blocks_3D_complex.x = prop.maxGridSize[0];
    }
    if(default_number_of_blocks_3D_complex.y > prop.maxGridSize[1]){
        default_number_of_blocks_3D_complex.y = prop.maxGridSize[1];
    }
    if(default_number_of_blocks_3D_complex.z > prop.maxGridSize[2]){
        default_number_of_blocks_3D_complex.z = prop.maxGridSize[2];
    }
}

