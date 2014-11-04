/**
 * @file        BaseLongMatrix.h
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file containing the base class for
 *              64b-wide integers (long for Linux/ size_t for Windows).
 *
 * @version     kspaceFirstOrder3D 3.3
 * @date        26 July     2011, 14:17  (created) \n
 *              04 November 2014, 17:11 (revised)
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

#ifndef BASE_LONG_MATRIX_H
#define BASE_LONG_MATRIX_H

#include "../MatrixClasses/BaseMatrix.h"
#include "../Utils/DimensionSizes.h"

using namespace std;

/**
 * @class TBaseLongMatrix
 * @brief Abstract base class for long based matrices defining basic interface.
 *        Higher dimensional matrices stored as 1D arrays, row-major order..
 */
class TBaseLongMatrix : public TBaseMatrix{
    public:

        TBaseLongMatrix(){
            pTotalElementCount = 0;
            pTotalAllocatedElementCount = 0;
            pDataRowSize = 0;
            p2DDataSliceSize = 0;
            pMatrixData = NULL;
            pdMatrixData = NULL;

        }

        /// Get dimension sizes of the matrix
        inline struct TDimensionSizes GetDimensionSizes() const {
            return pDimensionSizes;
        }

        /// Get total element count of the matrix
        virtual size_t GetTotalElementCount() const {
            return pTotalElementCount;
        };

        /// Get total allocated element count (might differ from total
        // element count used for the simulation because of padding).
        virtual size_t GetTotalAllocatedElementCount() const {
            return pTotalAllocatedElementCount;
        };

        /// Destructor
        virtual ~TBaseLongMatrix(){};

        //Host -> Host
        virtual void CopyData (TBaseLongMatrix& src);

        /// Get raw data out of the class (for direct kernel access).
        virtual size_t* GetRawData() {
            return pMatrixData;
        }


        virtual void SyncroniseToGPUDevice(){
            CopyIn(pMatrixData);
        }

        virtual void SyncroniseToCPUHost(){
            CopyOut(pMatrixData);
        }



        //Host -> Device
        virtual void CopyIn  (const size_t* HostSource);

        //Device -> Host
        virtual void CopyOut (size_t* HostDestination);

        //Device -> Device
        virtual void CopyForm(const size_t* DeviceSource);

        virtual size_t* GetRawDeviceData() {
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
        size_t* pMatrixData;


        /// Device matrix data
        size_t* pdMatrixData;


        /// Memory allocation
        virtual void AllocateMemory();

        /// Memory deallocation
        virtual void FreeMemory() ;

        /// Copy constructor is not directly allowed
        TBaseLongMatrix(const TBaseLongMatrix& orig);
        /// operator =  is not directly allowed
        TBaseLongMatrix & operator =(const TBaseLongMatrix&);

    private:

};// end of TBaseLongMatrix
//-----------------------------------------------------------------------------

#endif /* BASE_LONG_MATRIX_H */

