/**
 * @file        BaseFloatMatrix.cpp
 * @author      Jiri Jaros              \n
 *              CECS, ANU, Australia    \n
 *              jiri.jaros@anu.edu.au   \n
 *
 * @brief       The implementation file containing the base class for
 *              single precisions floating point numbers (floats)
 *
 * @version     kspaceFirstOrder3D 2.13
 * @date        11 July 2011, 12:13      (created) \n
 *              17 September 2012, 15:35 (revised)
 *
 * @section license
 * This file is part of the C++ extension of the k-Wave Toolbox
 * (http://www.k-wave.org).\n Copyright (C) 2012 Jiri Jaros and Bradley Treeby
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
 * along with k-Wave. If not, see <http://www.gnu.org/licenses/>.
 */


#ifdef __APPLE__
#include <stdlib.h>
#else
#include <malloc.h>
#endif

#include <string.h>
#include <assert.h>

#include "BaseFloatMatrix.h"
#include "../Parameters/Parameters.h"

#include "../Utils/DimensionSizes.h"
#include "../Utils/ErrorMessages.h"
#include "../Utils/MemoryMeasure.h"

#if VANILLA_CPP_VERSION

#endif
#if CUDA_VERSION
#include <cuda_runtime.h>
#include <iostream> //fprintf
#include <stdlib.h> //exit

using std::string;

inline void gpuAssert(cudaError_t code,
                      string file,
                      int line,
                      bool Abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,
                "GPUassert: %s %s %d\n",
                cudaGetErrorString(code),file.c_str(),line);
        if (Abort) exit(code);
    }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#endif
#if OPENCL_VERSION

#include "../OpenCL/ClSupremo.h"

#endif
//--------------------------------------------------------------------------//
//                            Constants                                     //
//--------------------------------------------------------------------------//

//--------------------------------------------------------------------------//
//                            Definitions                                   //
//--------------------------------------------------------------------------//

//--------------------------------------------------------------------------//
//                            Implementation                                //
//                            public methods                                //
//--------------------------------------------------------------------------//

/**
 *
 * Copy data from another matrix with same size.
 *
 * @param [in] src - source matrix
 *
 */
void TBaseFloatMatrix::CopyData(TBaseFloatMatrix & src)
{
    memcpy(pMatrixData,
           src.pMatrixData,
           sizeof(float)*pTotalAllocatedElementCount);
}//end of CopyDataSameSize
//----------------------------------------------------------------------------

#if CUDA_VERSION
/*
 *  Host   -> Device
 */
void TBaseFloatMatrix::CopyIn(const float * HostSource)
{
    gpuErrchk(cudaMemcpy(pdMatrixData,
                         HostSource,
                         pTotalAllocatedElementCount*sizeof(float),
                         cudaMemcpyHostToDevice));
}// end of CopyIn
//----------------------------------------------------------------------------

/*
 *  Device ->Host
 */
void TBaseFloatMatrix::CopyOut(float * HostDestination)
{
    gpuErrchk(cudaMemcpy(HostDestination,
                         pdMatrixData,
                         pTotalAllocatedElementCount*sizeof(float),
                         cudaMemcpyDeviceToHost));
}// end of CopyOut
//----------------------------------------------------------------------------

void TBaseFloatMatrix::CopyOut(float* HostDestination, size_t first_n_elements)
{
    gpuErrchk(cudaMemcpy(HostDestination,
                         pdMatrixData,
                         first_n_elements*sizeof(float),
                         cudaMemcpyDeviceToHost));
}

/*
 *   Device -> Device
 */
void TBaseFloatMatrix::CopyForm(const float * DeviceSource)
{
    gpuErrchk(cudaMemcpy(pdMatrixData,
                         DeviceSource,
                         pTotalAllocatedElementCount*sizeof(float),
                         cudaMemcpyDeviceToDevice));
}// end of CopyFrom
//----------------------------------------------------------------------------
#endif
#if OPENCL_VERSION
//Host   -> Device
void TBaseFloatMatrix::CopyIn(const float* HostSource)
{
    ClSupremo* handle = ClSupremo::GetInstance();
    handle->PassMemoryToClMemory(pdMatrixData,
                                 const_cast<float*>(HostSource),
                                 pTotalAllocatedElementCount);
}

//Device -> Host
void TBaseFloatMatrix::CopyOut(float* HostDestination)
{
    ClSupremo* handle = ClSupremo::GetInstance();
    handle->PassClMemoryToMemory(pdMatrixData,
                                 HostDestination,
                                 pTotalAllocatedElementCount);
}

//Device -> Host (but only the first n elements)
void TBaseFloatMatrix::CopyOut(float* HostDestination, size_t n)
{
    ClSupremo* handle = ClSupremo::GetInstance();
    handle->PassClMemoryToMemory(pdMatrixData,
                                 HostDestination,
                                 n);
}

//Device -> Device
void TBaseFloatMatrix::CopyForm(const cl_mem DeviceSource)
{
    ClSupremo* handle = ClSupremo::GetInstance();
    pdMatrixData = handle->CopyClMemory(DeviceSource,
                                        pTotalAllocatedElementCount);
}
#endif

//--------------------------------------------------------------------------//
//                            Implementation                                //
//                           protected methods                              //
//--------------------------------------------------------------------------//

/**
 * Memory allocation based on the total number of elements. \n
 * Memory is aligned by the SSE_ALIGNMENT and all elements are zeroed.
 */

void TBaseFloatMatrix::AllocateMemory(){
    assert(pMatrixData == NULL);

    //if testing, print the current memory size
    LogPreAlloc();

    //size of memory to allocate
    size_t size_in_bytes = pTotalAllocatedElementCount*sizeof(float);

    //if testing, print how much memory is requested
    LogAlloc(size_in_bytes);

    pMatrixData = static_cast<float*>(malloc(size_in_bytes));

    if (!pMatrixData) {
        fprintf(stderr,Matrix_ERR_FMT_Not_Enough_Memory, "TBaseFloatMatrix");
        throw bad_alloc();
    }

#if CUDA_VERSION
    assert(pdMatrixData == NULL);

    gpuErrchk(cudaMalloc((void **)&pdMatrixData,size_in_bytes));

    if (!pdMatrixData) {
        fprintf(stderr,Matrix_ERR_FMT_Not_Enough_Memory, "TBaseFloatMatrix");
        throw bad_alloc();
    }
#endif
#if OPENCL_VERSION
    assert(pdMatrixData == NULL);

    pdMatrixData =
        ClSupremo::GetInstance()->CreateClMemory(pTotalAllocatedElementCount);

    if (!pdMatrixData) {
        fprintf(stderr,Matrix_ERR_FMT_Not_Enough_Memory, "TBaseFloatMatrix");
        throw bad_alloc();
    }
#endif
    //if testing, print the current memory size
    LogPostAlloc();

}//end of AllocateMemory
//----------------------------------------------------------------------------

/**
 * Free memory
 */
void TBaseFloatMatrix::FreeMemory()
{

    if (pMatrixData){
        free(pMatrixData);
    }
    pMatrixData = NULL;

#if CUDA_VERSION
    if (pdMatrixData){
        gpuErrchk(cudaFree(pdMatrixData));
    }
    pdMatrixData = NULL;
#endif
#if OPENCL_VERSION
    if (pdMatrixData) clReleaseMemObject(pdMatrixData);
    pdMatrixData = NULL;
#endif

}//end of MemoryDealocation
//----------------------------------------------------------------------------

//--------------------------------------------------------------------------//
//                            Implementation                                //
//                            private methods                               //
//--------------------------------------------------------------------------//

