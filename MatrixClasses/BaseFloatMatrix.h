/**
 * @file        BaseFloatMatrix.h
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file containing the base class for
 *              single precisions floating point numbers (floats).
 *
 * @version     kspaceFirstOrder3D 3.3
 * @date        11 July      2011, 12:13 (created) \n
 *              04 November  2014, 17:11 (revised)
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

#ifndef BASEFLOATMATRIXDATA_H
#define	BASEFLOATMATRIXDATA_H

#include "BaseMatrix.h"
#include "../Utils/DimensionSizes.h"

using namespace std;

/**
 * @class TBaseFloatMatrix
 * @brief Abstract base class for float based matrices defining basic interface.
 *        Higher dimensional matrices stored as 1D arrays, row-major order.
 */
class TBaseFloatMatrix : public TBaseMatrix{
    public:

        TBaseFloatMatrix(){
            //TBaseMatrix();
            pTotalElementCount = 0;
            pTotalAllocatedElementCount = 0;
            //pDimensionSizes;
            pDataRowSize = 0;
            p2DDataSliceSize = 0;
            pMatrixData = NULL;
            pdMatrixData = NULL;

        };

        //Get dimension sizes of the matrix
        virtual TDimensionSizes GetDimensionSizes() const {
            return pDimensionSizes;
        }

        //Get total element count of the matrix
        virtual size_t GetTotalElementCount() const {
            return pTotalElementCount;
        };

        //Get total allocated element count (might differ from total element
        //count used for the simulation because of padding).
        virtual size_t GetTotalAllocatedElementCount() const {
            return pTotalAllocatedElementCount;
        };

        //Destructor
        virtual ~TBaseFloatMatrix() {};

        //Copy data from other matrix with the same size
        virtual void CopyData(TBaseFloatMatrix & src);

        //Get raw data out of the class (for direct kernel access).
        virtual float* GetRawData() {
            return pMatrixData;
        }


        virtual void SyncroniseToGPUDevice(){
            CopyIn(pMatrixData);
        }

        virtual void SyncroniseToCPUHost(){
            CopyOut(pMatrixData);
        }

        virtual void SyncroniseToCPUHost(size_t first_n_elements){
            CopyOut(pMatrixData,first_n_elements);
        }




        //Host   -> Device
        virtual void CopyIn  (const float * HostSource);

        //Device -> Host
        virtual void CopyOut (float * HostDestination);

        //Device -> Host (but only the first n elements)
        virtual void CopyOut(float* HostDestination, size_t n);

        //Device -> Device
        virtual void CopyForm(const float * DeviceSource);

        virtual float * GetRawDeviceData() {
            //copy pMatrixData to GPU Device memory (pdMatrixData)
            //CopyIn(pMatrixData);
            return pdMatrixData;
        }


    protected:

        /// Total number of elements
        size_t pTotalElementCount;
        /// Total number of allocated elements (the array size).
        size_t pTotalAllocatedElementCount;

        /// Dimension sizes
        struct TDimensionSizes pDimensionSizes;

        /// Size of 1D row in X dimension
        size_t pDataRowSize;
        /// Size of 2D slab (X,Y)
        size_t p2DDataSliceSize;

        /// Raw matrix data
        float* pMatrixData;

        /// Device matrix data
        float* pdMatrixData;


        //Memory allocation
        virtual void AllocateMemory();

        //Memory deallocation
        virtual void FreeMemory();

        //Copy constructor is not directly allowed
        TBaseFloatMatrix(const TBaseFloatMatrix& orig);
        //operator =  is not directly allowed
        TBaseFloatMatrix & operator =(const TBaseFloatMatrix&);

    private:

};//end of class TBaseFloatMatrix
//----------------------------------------------------------------------------

#endif /* TBASEMATRIXDATA_H */

