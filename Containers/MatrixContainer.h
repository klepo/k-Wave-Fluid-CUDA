/**
 * @file      MatrixContainer.h
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The header file containing the matrix container.
 *
 * @version   kspaceFirstOrder3D 3.6
 *
 * @date      02 December  2014, 16:17 (created) \n
 *            27 February  2019, 10:34 (revised)
 *
 * @copyright Copyright (C) 2019 Jiri Jaros and Bradley Treeby.
 *
 * This file is part of the C++ extension of the [k-Wave Toolbox](http://www.k-wave.org).
 *
 * This file is part of the k-Wave. k-Wave is free software: you can redistribute it and/or modify it under the terms
 * of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * k-Wave is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for
 * more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along with k-Wave.
 * If not, see [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/).
 */

#ifndef MATRIX_CONTAINER_H
#define MATRIX_CONTAINER_H

#include <map>

#include <MatrixClasses/BaseMatrix.h>
#include <MatrixClasses/BaseFloatMatrix.h>
#include <MatrixClasses/RealMatrix.h>
#include <MatrixClasses/ComplexMatrix.h>
#include <MatrixClasses/IndexMatrix.h>
#include <MatrixClasses/CufftComplexMatrix.h>

#include <Utils/MatrixNames.h>
#include <Utils/DimensionSizes.h>

#include <Containers/MatrixRecord.h>


/**
 * @class   MatrixContainer
 * @brief   Class implementing the matrix container.
 * @details This container is responsible to maintain all the matrices in the code except the output
 *          streams. The matrices are allocated, freed, loaded stored and check-pointed from here.
 */
class MatrixContainer
{
  public:

    /**
     * @enum  MatrixIdx
     * @brief Matrix identifers of all matrices in the 2D and 3D fluid k-space code.
     */
    enum class MatrixIdx
    {
      /// Kappa matrix.
      kKappa,
      /// Kappa for source scaling.
      kSourceKappa,
      /// c^2 matrix.
      kC2,
      /// Pressure matrix.
      kP,

      /// Acoustic density x.
      kRhoX,
      /// Acoustic density y.
      kRhoY,
      /// Acoustic density z.
      kRhoZ,

      /// Velocity x on staggered grid.
      kUxSgx,
      /// Velocity y on staggered grid.
      kUySgy,
      /// Velocity z on staggered grid.
      kUzSgz,

      /// Acoustic acceleration x.
      kDuxdx,
      /// Acoustic acceleration y.
      kDuydy,
      /// Acoustic acceleration z.
      kDuzdz,

      /// Initial velocity
      kRho0,
      /// dt / initial velocity on staggered grid x.
      kDtRho0Sgx,
      /// dt / initial velocity on staggered grid y.
      kDtRho0Sgy,
      /// dt / initial velocity on staggered grid z.
      kDtRho0Sgz,

      /// Positive Fourier shift in x.
      kDdxKShiftPosR,
      /// Positive Fourier shift in y.
      kDdyKShiftPos,
      /// Positive Fourier shift in z.
      kDdzKShiftPos,

      /// Negative Fourier shift in x
      kDdxKShiftNegR,
      /// Negative Fourier shift in y
      kDdyKShiftNeg,
      /// Negative Fourier shift in z
      kDdzKShiftNeg,

      /// PML on staggered grid x.
      kPmlXSgx,
      /// PML on staggered grid y.
      kPmlYSgy,
      /// PML on staggered grid z.
      kPmlZSgz,
      /// PML in x.
      kPmlX,
      /// PML in y.
      kPmlY,
      /// PML in z.
      kPmlZ,

      /// Nonlinear coefficient.
      kBOnA,
      /// Absorbing coefficient Tau.
      kAbsorbTau,
      /// Absorbing coefficient Eau.
      kAbsorbEta,
      /// Absorbing coefficient Nabla 1.
      kAbsorbNabla1,
      /// Absorbing coefficient Nabla 2.
      kAbsorbNabla2,

      /// Linear sensor mask.
      kSensorMaskIndex,
      /// Cuboid corners sensor mask.
      kSensorMaskCorners,

      /// Initial pressure source data.
      kInitialPressureSourceInput,
      /// Pressure source input data.
      kPressureSourceInput,
      /// Transducer source input data.
      kTransducerSourceInput,
      /// Velocity x source input data.
      kVelocityXSourceInput,
      /// Velocity y source input data.
      kVelocityYSourceInput,
      /// Velocity z source input data.
      kVelocityZSourceInput,
      /// Pressure source geometry data.
      kPressureSourceIndex,
      /// Velocity source geometry data.
      kVelocitySourceIndex,
      /// Delay mask for many types sources
      kDelayMask,

      /// Non uniform grid acoustic velocity in x.
      kDxudxn,
      /// Non uniform grid acoustic velocity in y.
      kDyudyn,
      /// Non uniform grid acoustic velocity in z.
      kDzudzn,
      /// Non uniform grid acoustic velocity on staggered grid x.
      kDxudxnSgx,
      /// Non uniform grid acoustic velocity on staggered grid y.
      kDyudynSgy,
      /// Non uniform grid acoustic velocity on staggered grid z.
      kDzudznSgz,


      /// velocity shift for non-staggered velocity in x.
      kUxShifted,
      /// velocity shift for non-staggered velocity in y.
      kUyShifted,
      /// velocity shift for non-staggered velocity in z.
      kUzShifted,

      /// Negative shift for non-staggered velocity in x.
      kXShiftNegR,
      /// Negative shift for non-staggered velocity in y.
      kYShiftNegR,
      /// Negative shift for non-staggered velocity in z.
      kZShiftNegR,

      /// 2D or 3D temporary matrix.
      kTemp1RealND,
      /// 2D or 3D temporary matrix.
      kTemp2RealND,
      /// 2D or 3D temporary matrix.
      kTemp3RealND,
      /// Temporary matrix for 1D fft in x.
      kTempCufftX,
      /// Temporary matrix for 1D fft in y.
      kTempCufftY,
      /// Temporary matrix for 1D fft in z.
      kTempCufftZ,
      /// Temporary matrix for cufft shift.
      kTempCufftShift
    };// end of MatrixIdx


    /// Constructor.
    MatrixContainer();
    /// Copy constructor is not allowed.
    MatrixContainer(const MatrixContainer&) = delete;
    /// Destructor.
    ~MatrixContainer();

    /// Operator = is not allowed.
    MatrixContainer& operator=(const MatrixContainer&) = delete;

    /**
     * @brief  Get the number of matrices in the container.
     * @return Number of matrices in the container.
     */
    inline size_t size() const
    {
      return mContainer.size();
    };

    /**
     * @brief  Is the container empty?
     * @return true - If the container is empty.
     */
    inline bool empty() const
    {
      return mContainer.empty();
    };

    /**
     * @brief  operator[]
     * @param  [in]  matrixIdx - Matrix identifier
     * @return Matrix record.
     */
    inline MatrixRecord& operator[](const MatrixIdx matrixIdx)
    {
      return mContainer[matrixIdx];
    };

    /**
     * @brief      Get the matrix with a specific type from the container.
     * @details    This template routine returns the reference to the matrix re-casted to the specific class type.
     * @param [in] matrixIdx - Matrix identifier,
     * @return     Reference to the Matrix,
     */
    template <typename T>
    inline T& getMatrix(const MatrixIdx matrixIdx)
    {
      return static_cast<T&> (*(mContainer[matrixIdx].matrixPtr));
    };

    /// Populate the container based on the simulation type.
    void init();

    /**
     * @brief Create all matrix objects in the container.
     * @throw std::bad_alloc        - Usually due to out of memory.
     * @throw std::invalid_argument - If this routine is called more than once.
     * @throw std::invalid_argument - If matrix type is unknown.
     */
    void createMatrices();
    /// Destroy and free all matrices.
    void freeMatrices();

    /// Load all marked matrices from the input HDF5 file.
    void loadDataFromInputFile();
    /// Load selected matrices from the checkpoint HDF5 file.
    void loadDataFromCheckpointFile();
    /// Store selected matrices into the checkpoint file.
    void storeDataIntoCheckpointFile();

    /// Copy all matrices from host to device (CPU -> GPU).
    void copyMatricesToDevice();
    /// Copy all matrices from device to host (GPU -> CPU).
    void copyMatricesFromDevice();

  protected:

  private:

    /// map holding the container
    std::map<MatrixIdx, MatrixRecord> mContainer;

};// end of MatrixContainer
//----------------------------------------------------------------------------------------------------------------------
#endif	/* MATRIX_CONTAINER_H */

