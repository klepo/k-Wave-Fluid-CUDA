/**
 * @file      MatrixContainer.h
 *
 * @author    Jiri Jaros, Petr Kleparnik \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The header file containing the matrix container.
 *
 * @version   kspaceFirstOrder 3.6
 *
 * @date      02 December  2014, 16:17 (created) \n
 *            08 February  2023, 12:00 (revised)
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
#include <Containers/MatrixRecord.h>
#include <MatrixClasses/BaseFloatMatrix.h>
#include <MatrixClasses/RealMatrix.h>
#include <MatrixClasses/ComplexMatrix.h>
#include <MatrixClasses/IndexMatrix.h>
#include <MatrixClasses/CufftComplexMatrix.h>

#include <Utils/MatrixNames.h>
#include <Utils/DimensionSizes.h>

/**
 * @class   MatrixContainer
 * @brief   Class implementing the matrix container.
 * @details This container is responsible to maintain all the matrices in the code except the output
 *          streams. The matrices are allocated, freed, loaded stored and check-pointed from here.
 *          The container data is set mutable in order to forbid adding and modifying MatrixRecords but allowing to
 *          modify matrix data by kernels.
 */
class MatrixContainer
{
  public:
    /**
     * @enum  MatrixIdx
     * @brief Matrix identifiers of all matrices in the 2D and 3D fluid k-space code.
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

      /// Velocity shift for non-staggered velocity in x.
      kUxShifted,
      /// Velocity shift for non-staggered velocity in y.
      kUyShifted,
      /// Velocity shift for non-staggered velocity in z.
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
    }; // end of MatrixIdx

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
     *         The const version is not offered since the container is mutable and one can modify records in the
     *         container.
     * @param  [in]  matrixIdx - Matrix identifier.
     * @return Matrix record.
     */
    inline MatrixRecord& operator[](const MatrixIdx matrixIdx)
    {
      return mContainer[matrixIdx];
    };

    /**
     * @brief      Get the matrix with a specific type from the container.
     * @details    This template routine returns the reference to the matrix re-casted to the specific class type.
     * @param [in] matrixIdx - Matrix identifier.
     * @return     Reference to the Matrix.
     */
    template<typename T> inline T& getMatrix(const MatrixIdx matrixIdx)
    {
      return static_cast<T&>(*(mContainer[matrixIdx].matrixPtr));
    }

    /**
     * @brief      Get the matrix with a specific type from the const container. The matrix is mutable
     * @details    This template routine returns the reference to the matrix re-casted to the specific class type.
     * @param [in] matrixIdx - Matrix identifier.
     * @return     Reference to the Matrix, which can be mutated.
     */
    template<typename T> inline T& getMatrix(const MatrixIdx matrixIdx) const
    {
      return static_cast<T&>(*(mContainer[matrixIdx].matrixPtr));
    }

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

    //----------------------------------------------- Get matrices ---------------------------------------------------//

    /**
     * @brief  Get the kappa matrix from the container.
     * @return Kappa matrix.
     */
    RealMatrix& getKappa() const
    {
      return getMatrix<RealMatrix>(MatrixIdx::kKappa);
    };

    /**
     * @brief  Get the sourceKappa matrix from the container.
     * @return Source kappa matrix.
     */
    RealMatrix& getSourceKappa() const
    {
      return getMatrix<RealMatrix>(MatrixIdx::kSourceKappa);
    };

    /**
     * @brief  Get the c^2 matrix from the container.
     * @return c^2 matrix.
     */
    RealMatrix& getC2() const
    {
      return getMatrix<RealMatrix>(MatrixIdx::kC2);
    };

    /**
     * @brief  Get pressure matrix.
     * @return Pressure matrix.
     */
    RealMatrix& getP() const
    {
      return getMatrix<RealMatrix>(MatrixIdx::kP);
    };

    //--------------------------------------------- Velocity matrices ------------------------------------------------//
    /**
     * @brief  Get velocity matrix on staggered grid in x direction.
     * @return Velocity matrix on staggered grid.
     */
    RealMatrix& getUxSgx() const
    {
      return getMatrix<RealMatrix>(MatrixIdx::kUxSgx);
    };

    /**
     * @brief  Get velocity matrix on staggered grid in y direction.
     * @return Velocity matrix on staggered grid.
     */
    RealMatrix& getUySgy() const
    {
      return getMatrix<RealMatrix>(MatrixIdx::kUySgy);
    };

    /**
     * @brief  Get velocity matrix on staggered grid in z direction.
     * @return Velocity matrix on staggered grid.
     */
    RealMatrix& getUzSgz() const
    {
      return getMatrix<RealMatrix>(MatrixIdx::kUzSgz);
    };

    /**
     * @brief  Get velocity shifted on normal grid in x direction.
     * @return Unstaggeted velocity matrix.
     */
    RealMatrix& getUxShifted() const
    {
      return getMatrix<RealMatrix>(MatrixIdx::kUxShifted);
    };

    /**
     * @brief  Get velocity shifted on normal grid in y direction.
     * @return Unstaggered velocity matrix.
     */
    RealMatrix& getUyShifted() const
    {
      return getMatrix<RealMatrix>(MatrixIdx::kUyShifted);
    };

    /**
     * @brief  Get velocity shifted on normal grid in z direction.
     * @return Unstaggered velocity matrix.
     */
    RealMatrix& getUzShifted() const
    {
      return getMatrix<RealMatrix>(MatrixIdx::kUzShifted);
    };

    //----------------------------------------- Velocity gradient matrices -------------------------------------------//
    /**
     * @brief  Get velocity gradient on in x direction.
     * @return Velocity gradient matrix.
     */
    RealMatrix& getDuxdx() const
    {
      return getMatrix<RealMatrix>(MatrixIdx::kDuxdx);
    };

    /**
     * @brief  Get velocity gradient on in y direction.
     * @return Velocity gradient matrix.
     */
    RealMatrix& getDuydy() const
    {
      return getMatrix<RealMatrix>(MatrixIdx::kDuydy);
    };

    /**
     * @brief  Get velocity gradient on in z direction.
     * @return Velocity gradient matrix.
     */
    RealMatrix& getDuzdz() const
    {
      return getMatrix<RealMatrix>(MatrixIdx::kDuzdz);
    };

    //---------------------------------------------- Density matrices ------------------------------------------------//
    /**
     * @brief  Get dt * rho0Sgx matrix (time step size * ambient velocity on staggered grid in x direction).
     * @return Density matrix
     */
    RealMatrix& getDtRho0Sgx() const
    {
      return getMatrix<RealMatrix>(MatrixIdx::kDtRho0Sgx);
    };

    /**
     * @brief  Get dt * rho0Sgy matrix (time step size * ambient velocity on staggered grid in y direction).
     * @return Density matrix
     */
    RealMatrix& getDtRho0Sgy() const
    {
      return getMatrix<RealMatrix>(MatrixIdx::kDtRho0Sgy);
    };

    /**
     * @brief  Get dt * rho0Sgz matrix (time step size * ambient velocity on staggered grid in z direction).
     * @return Density matrix
     */
    RealMatrix& getDtRho0Sgz() const
    {
      return getMatrix<RealMatrix>(MatrixIdx::kDtRho0Sgz);
    };

    /**
     * @brief  Get density matrix in x direction.
     * @return Density matrix.
     */
    RealMatrix& getRhoX() const
    {
      return getMatrix<RealMatrix>(MatrixIdx::kRhoX);
    };

    /**
     * @brief  Get density matrix in y direction.
     * @return Density matrix.
     */
    RealMatrix& getRhoY() const
    {
      return getMatrix<RealMatrix>(MatrixIdx::kRhoY);
    };

    /**
     * @brief  Get density matrix in z direction.
     * @return Density matrix.
     */
    RealMatrix& getRhoZ() const
    {
      return getMatrix<RealMatrix>(MatrixIdx::kRhoZ);
    };

    /**
     * @brief  Get ambient density matrix.
     * @return Density matrix.
     */
    RealMatrix& getRho0() const
    {
      return getMatrix<RealMatrix>(MatrixIdx::kRho0);
    };

    //----------------------------------------------- Shift matrices -------------------------------------------------//
    /**
     * @brief  Get positive Fourier shift in x.
     * @return Shift matrix.
     */
    ComplexMatrix& getDdxKShiftPos() const
    {
      return getMatrix<ComplexMatrix>(MatrixIdx::kDdxKShiftPosR);
    };

    /**
     * @brief  Get positive Fourier shift in y.
     * @return Shift matrix.
     */
    ComplexMatrix& getDdyKShiftPos() const
    {
      return getMatrix<ComplexMatrix>(MatrixIdx::kDdyKShiftPos);
    };

    /**
     * @brief  Get positive Fourier shift in z.
     * @return Shift matrix.
     */
    ComplexMatrix& getDdzKShiftPos() const
    {
      return getMatrix<ComplexMatrix>(MatrixIdx::kDdzKShiftPos);
    };

    /**
     * @brief  Get negative Fourier shift in x.
     * @return Shift matrix.
     */
    ComplexMatrix& getDdxKShiftNeg() const
    {
      return getMatrix<ComplexMatrix>(MatrixIdx::kDdxKShiftNegR);
    };

    /**
     * @brief  Get negative Fourier shift in y.
     * @return Shift matrix.
     */
    ComplexMatrix& getDdyKShiftNeg() const
    {
      return getMatrix<ComplexMatrix>(MatrixIdx::kDdyKShiftNeg);
    };

    /**
     * @brief  Get negative Fourier shift in z.
     * @return Shift matrix.
     */
    ComplexMatrix& getDdzKShiftNeg() const
    {
      return getMatrix<ComplexMatrix>(MatrixIdx::kDdzKShiftNeg);
    };

    /**
     * @brief  Get negative shift for non-staggered velocity in x.
     * @return Shift matrix.
     */
    ComplexMatrix& getXShiftNegR() const
    {
      return getMatrix<ComplexMatrix>(MatrixIdx::kXShiftNegR);
    };

    /**
     * @brief  Get negative shift for non-staggered velocity in y.
     * @return Shift matrix.
     */
    ComplexMatrix& getYShiftNegR() const
    {
      return getMatrix<ComplexMatrix>(MatrixIdx::kYShiftNegR);
    };

    /**
     * @brief  Get negative shift for non-staggered velocity in z.
     * @return Shift matrix.
     */
    ComplexMatrix& getZShiftNegR() const
    {
      return getMatrix<ComplexMatrix>(MatrixIdx::kZShiftNegR);
    };

    //------------------------------------------------ PML matrices --------------------------------------------------//
    /**
     * @brief  Get PML on staggered grid x.
     * @return PML matrix.
     */
    RealMatrix& getPmlXSgx() const
    {
      return getMatrix<RealMatrix>(MatrixIdx::kPmlXSgx);
    };

    /**
     * @brief  Get PML on staggered grid y.
     * @return PML matrix.
     */
    RealMatrix& getPmlYSgy() const
    {
      return getMatrix<RealMatrix>(MatrixIdx::kPmlYSgy);
    };

    /**
     * @brief  Get PML on staggered grid z.
     * @return PML matrix.
     */
    RealMatrix& getPmlZSgz() const
    {
      return getMatrix<RealMatrix>(MatrixIdx::kPmlZSgz);
    };

    /**
     * @brief  Get PML in x.
     * @return PML matrix.
     */
    RealMatrix& getPmlX() const
    {
      return getMatrix<RealMatrix>(MatrixIdx::kPmlX);
    };

    /**
     * @brief  Get PML in y.
     * @return PML matrix.
     */
    RealMatrix& getPmlY() const
    {
      return getMatrix<RealMatrix>(MatrixIdx::kPmlY);
    };

    /**
     * @brief  Get PML in z.
     * @return PML matrix.
     */
    RealMatrix& getPmlZ() const
    {
      return getMatrix<RealMatrix>(MatrixIdx::kPmlZ);
    };

    //------------------------------------------- Nonlinear grid matrices --------------------------------------------//
    /**
     * @brief  Non uniform grid acoustic velocity in x.
     * @return Velocity matrix.
     */
    RealMatrix& getDxudxn() const
    {
      return getMatrix<RealMatrix>(MatrixIdx::kDxudxn);
    };

    /**
     * @brief  Non uniform grid acoustic velocity in y.
     * @return Velocity matrix.
     */
    RealMatrix& getDyudyn() const
    {
      return getMatrix<RealMatrix>(MatrixIdx::kDyudyn);
    };

    /**
     * @brief  Non uniform grid acoustic velocity in z.
     * @return Velocity matrix.
     */
    RealMatrix& getDzudzn() const
    {
      return getMatrix<RealMatrix>(MatrixIdx::kDzudzn);
    };

    /**
     * @brief  Non uniform grid acoustic velocity on staggered grid x.
     * @return Velocity matrix.
     */
    RealMatrix& getDxudxnSgx() const
    {
      return getMatrix<RealMatrix>(MatrixIdx::kDxudxnSgx);
    };

    /**
     * @brief  Non uniform grid acoustic velocity on staggered grid x.
     * @return Velocity matrix.
     */
    RealMatrix& getDyudynSgy() const
    {
      return getMatrix<RealMatrix>(MatrixIdx::kDyudynSgy);
    };

    /**
     * @brief  Non uniform grid acoustic velocity on staggered grid x.
     * @return Velocity matrix.
     */
    RealMatrix& getDzudznSgz() const
    {
      return getMatrix<RealMatrix>(MatrixIdx::kDzudznSgz);
    };

    //-------------------------------------- Nonlinear and absorption matrices ---------------------------------------//
    /**
     * @brief  Get B on A (nonlinear coefficient).
     * @return Nonlinear coefficient.
     */
    RealMatrix& getBOnA() const
    {
      return getMatrix<RealMatrix>(MatrixIdx::kBOnA);
    };

    /**
     * @brief  Get absorbing coefficient Tau.
     * @return Absorbing coefficient.
     */
    RealMatrix& getAbsorbTau() const
    {
      return getMatrix<RealMatrix>(MatrixIdx::kAbsorbTau);
    };

    /**
     * @brief  Get absorbing coefficient Eta.
     * @return Absorbing coefficient.
     */
    RealMatrix& getAbsorbEta() const
    {
      return getMatrix<RealMatrix>(MatrixIdx::kAbsorbEta);
    };

    /**
     * @brief  Get absorbing coefficient Nabla1.
     * @return Absorbing coefficient.
     */
    RealMatrix& getAbsorbNabla1() const
    {
      return getMatrix<RealMatrix>(MatrixIdx::kAbsorbNabla1);
    };

    /**
     * @brief  Get absorbing coefficient Nabla2.
     * @return Absorbing coefficient.
     */
    RealMatrix& getAbsorbNabla2() const
    {
      return getMatrix<RealMatrix>(MatrixIdx::kAbsorbNabla2);
    };

    //----------------------------------------------- Index matrices -------------------------------------------------//
    /**
     * @brief  Get linear sensor mask (spatial geometry of the sensor).
     * @return Sensor mask data.
     */
    IndexMatrix& getSensorMaskIndex() const
    {
      return getMatrix<IndexMatrix>(MatrixIdx::kSensorMaskIndex);
    };

    /**
     * @brief  Get cuboid corners sensor mask. (Spatial geometry of multiple sensors).
     * @return Sensor mask data.
     */
    IndexMatrix& getSensorMaskCorners() const
    {
      return getMatrix<IndexMatrix>(MatrixIdx::kSensorMaskCorners);
    };

    /**
     * @brief  Get velocity source geometry data.
     * @return Source geometry indices
     */
    IndexMatrix& getVelocitySourceIndex() const
    {
      return getMatrix<IndexMatrix>(MatrixIdx::kVelocitySourceIndex);
    };

    /**
     * @brief  Get pressure source geometry data.
     * @return Source geometry indices
     */
    IndexMatrix& getPressureSourceIndex() const
    {
      return getMatrix<IndexMatrix>(MatrixIdx::kPressureSourceIndex);
    };

    /**
     * @brief  Get delay mask for many types sources
     * @return delay mask.
     */
    IndexMatrix& getDelayMask() const
    {
      return getMatrix<IndexMatrix>(MatrixIdx::kDelayMask);
    }

    //-------------------------------------------------- Sources  ----------------------------------------------------//
    /**
     * @brief  Get transducer source input data (signal).
     * @return Transducer source input data.
     */
    RealMatrix& getTransducerSourceInput() const
    {
      return getMatrix<RealMatrix>(MatrixIdx::kTransducerSourceInput);
    };

    /**
     * @brief  Get pressure source input data (signal).
     * @return Pressure source input data.
     */
    RealMatrix& getPressureSourceInput() const
    {
      return getMatrix<RealMatrix>(MatrixIdx::kPressureSourceInput);
    };

    /**
     * @brief  Get initial pressure source input data (whole matrix).
     * @return Initial pressure source input data.
     */
    RealMatrix& getInitialPressureSourceInput() const
    {
      return getMatrix<RealMatrix>(MatrixIdx::kInitialPressureSourceInput);
    };

    /**
     * @brief  Get Velocity source input data in x direction.
     * @return Velocity source input data.
     */
    RealMatrix& getVelocityXSourceInput() const
    {
      return getMatrix<RealMatrix>(MatrixIdx::kVelocityXSourceInput);
    };

    /**
     * @brief  Get Velocity source input data in y direction.
     * @return Velocity source input data.
     */
    RealMatrix& getVelocityYSourceInput() const
    {
      return getMatrix<RealMatrix>(MatrixIdx::kVelocityYSourceInput);
    };

    /**
     * @brief  Get Velocity source input data in z direction.
     * @return Velocity source input data.
     */
    RealMatrix& getVelocityZSourceInput() const
    {
      return getMatrix<RealMatrix>(MatrixIdx::kVelocityZSourceInput);
    };

    //--------------------------------------------- Temporary matrices -----------------------------------------------//
    /**
     * @brief  Get first real 2D/3D temporary matrix.
     * @return Temporary real 2D/3D matrix.
     */
    RealMatrix& getTemp1RealND() const
    {
      return getMatrix<RealMatrix>(MatrixIdx::kTemp1RealND);
    };

    /**
     * @brief  Get second real 2D/3D temporary matrix.
     * @return Temporary real 2D/3D matrix.
     */
    RealMatrix& getTemp2RealND() const
    {
      return getMatrix<RealMatrix>(MatrixIdx::kTemp2RealND);
    };

    /**
     * @brief  Get third real 2D/3D temporary matrix.
     * @return Temporary real 2D/3D matrix.
     */
    RealMatrix& getTemp3RealND() const
    {
      return getMatrix<RealMatrix>(MatrixIdx::kTemp3RealND);
    };

    /**
     * @brief  Get temporary matrix for 1D fft in x.
     * @return Temporary complex 2D/3D matrix.
     */
    CufftComplexMatrix& getTempCufftX() const
    {
      return getMatrix<CufftComplexMatrix>(MatrixIdx::kTempCufftX);
    };

    /**
     * @brief  Get temporary matrix for 1D fft in y.
     * @return Temporary complex 2D/3D matrix.
     */
    CufftComplexMatrix& getTempCufftY() const
    {
      return getMatrix<CufftComplexMatrix>(MatrixIdx::kTempCufftY);
    };

    /**
     * @brief  Get temporary matrix for 1D fft in z.
     * @return Temporary complex 3D matrix.
     */
    CufftComplexMatrix& getTempCufftZ() const
    {
      return getMatrix<CufftComplexMatrix>(MatrixIdx::kTempCufftZ);
    };

    /**
     * @brief  Get temporary matrix for cufft shift.
     * @return Temporary complex 3D matrix.
     */
    CufftComplexMatrix& getTempCufftShift() const
    {
      return getMatrix<CufftComplexMatrix>(MatrixIdx::kTempCufftShift);
    };

  protected:
  private:
    /// map holding the container, it is mutable since we want to modify data in matrices within the container.
    mutable std::map<MatrixIdx, MatrixRecord> mContainer;

}; // end of MatrixContainer

//----------------------------------------------------------------------------------------------------------------------
#endif /* MATRIX_CONTAINER_H */
