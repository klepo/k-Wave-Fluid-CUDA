/**
 * @file        Parameters.h
 *
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file containing the parameters of the simulation.
 *
 * @version     kspaceFirstOrder3D 3.4
 *
 * @date        08 December 2011, 16:34 (created)      \n
 *              08 August   2017, 13:41 (revised)
 *
 * @section License
 * This file is part of the C++ extension of the k-Wave Toolbox
 * (http://www.k-wave.org).\n Copyright (C) 2016 Jiri Jaros and Bradley Treeby.
 *
 * This file is part of the k-Wave. k-Wave is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later version.
 *
 * k-Wave is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser
 * General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along with k-Wave.
 * If not, see http://www.gnu.org/licenses/.
 */

#ifndef ParametersH
#define ParametersH

#include <string>

#include <Parameters/CudaParameters.h>
#include <Parameters/CommandLineParameters.h>
#include <Utils/DimensionSizes.h>
#include <Hdf5/Hdf5FileHeader.h>
#include <Hdf5/Hdf5FileHeader.h>

/**
 * @class   Parameters
 * @brief   Class storing all parameters of the simulation.
 * @details Class storing all parameters of the simulation.
 * @warning This is a singleton class.
 *
 */
class Parameters
{
  public:

    /**
     * @enum    SensorMaskType
     * @brief   Sensor mask type (linear or cuboid corners).
     * @details Sensor mask type (linear or cuboid corners).
     */
    enum class SensorMaskType
    {
      /// Linear sensor mask
      kIndex   = 0,
      /// Cuboid corners sensor mask
      kCorners = 1
    };

    /// Copy constructor not allowed
    Parameters(const Parameters&) = delete;
    /// Destructor.
    virtual ~Parameters();
    /// operator= not allowed.
    Parameters& operator=(const Parameters&) = delete;

    /**
     * @brief  Get instance of the singleton class
     * @return The only instance of the class.
     */
    static Parameters& getInstance();

    /**
     * @brief Parse command line and read scalar values form the input file.
     * @param [in] argc - Number of commandline parameters
     * @param [in] argv - Commandline parameters
     * @throw std::invalid_argument - If sampling is supposed to start out of the simulation time span.
     */
    void init(int argc, char** argv);

    /// Print the simulation setup (all parameters).
    void printSimulatoinSetup();
    /**
     * @brief  Shall the code print version and exit?
     * @return True if the flag is set.
     */
    bool isPrintVersionOnly() const {return mCommandLineParameters.isPrintVersionOnly();};

    /**
     * @brief  Read scalar values from the input HDF5 file.
     * @throw ios:failure           - If the file cannot be open or is of a wrong type or version.
     * @throw std::invalid_argument - If some values are not correct or not supported.
     * @warning The file is opened in this routine and left open for further use.
     */
    void readScalarsFromInputFile();
    /**
     * @brief Save scalar values into the output HDF5 file.
     * @throw ios:failure - If the output file is closed.
     * @warning The file is expected to be open.
     */
    void saveScalarsToOutputFile();

    /**
     * @brief Get git hash of the code
     * @return Git hash compiled in using -D parameter
     */
    std::string getGitHash() const;

    /**
     * @brief Select Cuda device for execution.
     * @throw std::runtime_error - If there is no free CUDA devices.
     * @throw std::runtime_error - If there is no device of such and deviceIdx.
     * @throw std::runtime_error - If the GPU chosen is not supported (i.e., the code was not compiled for its
     *                             architecture)
     */
    void selectDevice();

    /**
     * @brief  Get class with CudaParameters (runtime setup), const version.
     * @return CudaParameters class.
     */
    const CudaParameters& getCudaParameters() const {return mCudaParameters;};

    /**
     * @brief  Get class with CudaParameters (runtime setup), mutable version.
     * @return CudaParameters class.
     */
    CudaParameters&       getCudaParameters()       {return mCudaParameters;};


    /**
     * @brief Get number of CPU threads to use
     * @return Number of CPU threads to use
     */
    size_t getNumberOfThreads()        const {return mCommandLineParameters.getNumberOfThreads();};

    /**
     * @brief  Get compression level.
     * @return Compression level value for output and checkpoint files.
     */
    size_t getCompressionLevel()       const {return mCommandLineParameters.getCompressionLevel();};

    /**
     * @brief Get progress print interval.
     * @return How often to print progress.
     */
    size_t getProgressPrintInterval()  const {return mCommandLineParameters.getProgressPrintInterval();};


    /**
     * @brief  Is checkpoint enabled?
     * @return true if checkpointing is enabled.
     */
    bool   isCheckpointEnabled()        const {return mCommandLineParameters.isCheckpointEnabled();};
    /**
     * @brief  Get checkpoint interval.
     * @return Checkpoint interval in seconds.
     */
    size_t getCheckpointInterval()      const {return mCommandLineParameters.getCheckpointInterval();};

    //---------------------------------------------------- Files -----------------------------------------------------//
    /**
     * @brief Get input file handle.
     * @return Handle to the input file.
     */
    Hdf5File& getInputFile()                 {return mInputFile;};
    /**
     * @brief Get output file handle.
     * @return Handle to the output file.
     */
    Hdf5File& getOutputFile()                {return mOutputFile;};
    /**
     * @brief Get checkpoint file handle.
     * @return Handle to the checkpoint file.
     */
    Hdf5File& getCheckpointFile()            {return mCheckpointFile;};
    /**
     * @brief Get file header handle.
     * @return Handle to the file header.
     */
    Hdf5FileHeader& getFileHeader()          {return mFileHeader;};


    /**
     * @brief  Get input file name.
     * @return Input file name
     */
    std::string getInputFileName()      const {return mCommandLineParameters.getInputFileName();};
    /**
     * @brief  Get output file name.
     * @return Output file name.
     */
    std::string getOutputFileName()     const {return mCommandLineParameters.getOutputFileName();};
    /**
     * @brief  Get checkpoint file name.
     * @return Checkpoint file name
     */
    std::string getCheckpointFileName() const {return mCommandLineParameters.getCheckpointFileName();};



    //-------------------------------------------- Simulation dimensions ---------------------------------------------//
    /**
     * @brief  Get full dimension sizes of the simulation (real classes).
     * @return Dimension sizes of 3D real matrices.
     */
    DimensionSizes getFullDimensionSizes()    const {return mFullDimensionSizes; };
    ///
    /**
     * @brief  Get reduced dimension sizes of the simulation (complex classes).
     * @return Dimension sizes of reduced complex 3D matrices
     */
    DimensionSizes getReducedDimensionSizes() const {return mReducedDimensionSizes;};

    /**
     * @brief  Get total number of time steps.
     * @return Total number of time steps.
     */
    size_t  getNt()                           const {return mNt;};

    /**
     * @brief  Get actual simulation time step.
     * @return Actual time step.
     */
    size_t  getTimeIndex()                    const {return mTimeIndex;};

    /**
     * @brief  Set simulation time step - should be used only when recovering from checkpoint
     * @param [in] timeIndex - Actual time step.
     */
    void setTimeIndex(const size_t timeIndex)       {mTimeIndex = timeIndex;};
    /// Increment simulation time step.
    void incrementTimeIndex()                       {mTimeIndex++;};

    //----------------------------------------------- Grid parameters ------------------------------------------------//
    /**
     * @brief  Get time step size.
     * @return dt value.
     */
    float getDt() const {return mDt;};
    /**
     * @brief  Get spatial displacement in x.
     * @return dx value.
     */
    float getDx() const {return mDx;};
    /**
     * @brief  Get spatial displacement in y.
     * @return dy value.
     */
    float getDy() const {return mDy;};
    /**
     * @brief  Get spatial displacement in z.
     * @return dz value
     */
    float getDz() const {return mDz;};

    //------------------------------------------- Sound speed and density --------------------------------------------//
    /**
     * @brief  Get reference sound speed.
     * @return Reference sound speed.
     */
    float  getCRef()            const {return mCRef;};
    /**
     * @brief  Is sound speed in the medium homogeneous (scalar value)?
     * @return True if scalar
     */
    bool   getC0ScalarFlag()    const {return mC0ScalarFlag;};
    /**
     * @brief  Get scalar value of sound speed.
     * @return Sound speed.
     */
    float  getC0Scalar()        const {return mC0Scalar;};
   /**
     * @brief  Get scalar value of sound speed squared.
     * @return Sound speed.
     */
    float  getC2Scalar()        const {return mC0Scalar * mC0Scalar;};

    /**
     * @brief  Is density in the medium homogeneous (scalar value)?
     * @return True if scalar
     */
    bool   getRho0ScalarFlag()  const {return mRho0ScalarFlag;};
    /**
     * @brief  Get value of homogeneous medium density.
     * @return Density.
     */
    float  getRho0Scalar()      const {return mRho0Scalar;};
    /**
     * @brief  Get value of homogeneous medium density on staggered grid in x direction.
     * @return Staggered density.
     */
    float  getRho0SgxScalar()   const {return mRho0SgxScalar;};
    /**
     * @brief  Get value of dt / rho0Sgx.
     * @return Staggered density.
     */
    float  getDtRho0SgxScalar() const {return mDt / mRho0SgxScalar;};
    /**
     * @brief  Get value of homogeneous medium density on staggered grid in y direction.
     * @return Staggered density.
     */
    float  getRho0SgyScalar()   const {return mRho0SgyScalar;};
    /**
     * @brief  Get value of dt / rho0Sgy.
     * @return Staggered density.
     */
    float  getDtRho0SgyScalar() const {return mDt / mRho0SgyScalar;};
    /**
     * @brief  Get value of homogeneous medium density on staggered grid in z direction.
     * @return Staggered density.
     */
    float  getRho0SgzScalar()   const {return mRho0SgzScalar;};
    /**
     * @brief  Get value of dt / rho0Sgz.
     * @return Staggered density.
     */
    float  getDtRho0SgzScalar() const {return mDt / mRho0SgzScalar;};

    //----------------------------------------- Absorption and nonlinearity ------------------------------------------//
    /**
     * @brief  Enable non uniform grid? - not implemented yet.
     * @return Non uniform flag
     */
    size_t getNonUniformGridFlag()   const {return mNonUniformGridFlag;};
    /**
     * @brief  Is the simulation absrobing or lossless?
     * @return 0 if the medium is lossless, 1 otherwise.
     */
    size_t getAbsorbingFlag()        const {return mAbsorbingFlag;};
    /**
     * @brief  Is the wave propagation nonlinear?
     * @return 0 if the simulation is linear, 1 otherwise.
     */
    size_t getNonLinearFlag()        const {return mNonLinearFlag;};


    /**
     * @brief  Is alpha absorption coefficient homogeneous (scalar value)?
     * @return True if scalar
     */
    bool   getAlphaCoeffScalarFlag() const {return mAlphaCoeffScalarFlag;};
    /**
     * @brief  Get value of alpha absorption coefficient.
     * @return Alpha absorption coefficient.
     */
    float  getAlphaCoeffScalar()     const {return mAlphaCoeffScalar; }
    /**
     * @brief  Get alpha power value for the absorption law.
     * @return Alpha power value.
     */
    float  getAlphaPower()           const {return mAlphaPower;};
    /**
     * @brief  Get absorb eta coefficient for homogeneous medium (scalar value)?
     * @return Absorb eta coefficient.
     */
    float  getAbsorbEtaScalar()      const {return mAbsorbEtaScalar;};
    /**
     * @brief Set absorb eta coefficient for homogeneous medium (scalar value).
     * @param [in] absrobEta - New value for absorb eta
     */
    void   setAbsorbEtaScalar(const float absrobEta) {mAbsorbEtaScalar = absrobEta;};
    /**
     * @brief  Get absorb tau coefficient for homogeneous medium.
     * @return Absorb tau coefficient.
     */
    float  getAbsorbTauScalar()      const {return mAbsorbTauScalar;};
    /**
     * @brief Set absorb tau coefficient for homogeneous medium (scalar value)
     * @param [in] absorbTau - New value for absorb tau.
     */
    void   setAbsorbTauScalar(const float absorbTau) {mAbsorbTauScalar = absorbTau;};


    /**
     * @brief  Is nonlinear coefficient homogeneous in the medium (scalar value)?
     * @return True if scalar
     */
    bool   getBOnAScalarFlag()       const {return mBOnAScalarFlag;};
    /**
     * @brief  Get nonlinear coefficient for homogenous medium.
     * @return Nonlinear coefficient.
     */
    float  getBOnAScalar()           const {return mBOnAScalar;};


    //------------------------------------------ Perfectly matched layer ---------------------------------------------//
    /**
     * @brief  Get depth of the perfectly matched layer in x.
     * @return PML size in x
     */
    size_t getPmlXSize()  const {return mPmlXSize;};
    /**
     * @brief  Get depth of the perfectly matched layer in y.
     * @return PML size in y.
     */
    size_t getPmlYSize()  const {return mPmlYSize;};
    /**
     * @brief  Get depth of the perfectly matched layer in z.
     * @return PML size in z.
     */
    size_t getPmlZSize()  const {return mPmlZSize;};

    /**
     * @brief  Get Perfectly matched layer attenuation in x, not implemented.
     * @return Attenuation for PML in x.
     */
    float  getPmlXAlpha() const {return mPmlXAlpha;};
    /**
     * @brief  Get Perfectly matched layer attenuation in y, not implemented.
     * @return Attenuation for PML in y.
     */
    float  getPmlYAlpha() const {return mPmlYAlpha;};
    /**
     * @brief  Get Perfectly matched layer attenuation in z , not implemented.
     * @return Attenuation for PML in z.
     */
    float  getPmlZAlpha() const {return mPmlZAlpha;};

    //-------------------------------------------------- Sources -----------------------------------------------------//
    /**
     * @brief  Get pressure source flag.
     * @return 0 if the source is disabled.
     * @return Length of the input signal in time steps.
     */
    size_t getPressureSourceFlag()        const {return mPressureSourceFlag; };
    /**
     * @brief  Get initial pressure source flag (p0).
     * @return 0 if the source is disabled, 1 otherwise.
     */
    size_t getInitialPressureSourceFlag() const {return mInitialPressureSourceFlag;};
    /**
     * @brief  Get transducer source flag.
     * @return 0 if the transducer is disabled.
     * @return Length of the input signal in time steps.
     */
    size_t getTransducerSourceFlag()      const {return mTransducerSourceFlag;};
    /**
     * @brief  Get velocity in x source flag.
     * @return 0 if the source is disabled.
     * @return Length of the input signal in time steps.
     */
    size_t getVelocityXSourceFlag()       const {return mVelocityXSourceFlag;};
    /**
     * @brief  Get velocity in y source flag.
     * @return 0 if the source is disabled.
     * @return Length of the input signal in time steps.
     */
    size_t getVelocityYSourceFlag()       const {return mVelocityYSourceFlag;};
    /**
     * @brief  Get velocity in z source flag.
     * @return 0 if the source is disabled.
     * @return Length of the input signal in time steps.
     */
    size_t getVelocityZSourceFlag()       const {return mVelocityZSourceFlag;};


    /**
     * @brief  Get spatial size of the pressure source.
     * @return Size of the pressure source in grid points.
     */
    size_t getPressureSourceIndexSize()   const { return mPressureSourceIndexSize;}
    /**
     * @brief  Get spatial size of the transducer source.
     * @return Size of the transducer source in grid points.
     */
    size_t getTransducerSourceInputSize() const { return mTransducerSourceInputSize;}
    /**
     * @brief  Get spatial size of the velocity source.
     * @return Size of the velocity source in grid points.
     */
    size_t getVelocitySourceIndexSize()   const { return mVelocitySourceIndexSize;}


    /**
     * @brief  Get pressure source mode.
     * @return Pressure source mode (Dirichlet or additive).
     */
    size_t getPressureSourceMode()        const {return mPressureSourceMode;};
    /**
     * @brief  Get number of time series in the pressure source.
     * @return Number of time series in the pressure source
     */
    size_t getPressureSourceMany()        const {return mPressureSourceMany;};

    /**
     * @brief  Get velocity source mode.
     * @return Pressure source mode (Dirichlet or additive).
     */
    size_t getVelocitySourceMode()        const {return mVelocitySourceMode;};
    /**
     * @brief  Get number of time series in the velocity sources.
     * @return Number of time series in the velocity sources.
     */
    size_t  getVelocitySourceMany()       const {return mVelocitySourceMany;};



    //-------------------------------------------------- Sensors -----------------------------------------------------//
    /**
     * @brief  Get sensor mask type (linear or corners).
     * @return Sensor mask type.
     */
    SensorMaskType getSensorMaskType()     const {return mSensorMaskType;};
    /**
     * @brief  Get spatial size of the index sensor mask.
     * @return Number of grid points.
     */
    size_t getSensorMaskIndexSize()        const {return mSensorMaskIndexSize;}
    /**
     * @brief  Get number of cuboids the sensor is composed of.
     * @return Number of cuboids.
     */
    size_t getSensorMaskCornersSize()      const {return mSensorMaskCornersSize;};
    /**
     * @brief  Get start time index when sensor data collection begins
     * @return When to start sampling data.
     */
    size_t getSamplingStartTimeIndex()     const {return mCommandLineParameters.getSamplingStartTimeIndex();};


    /**
     * @brief  Is  -p or --p_raw specified at the command line?
     * @return True if the flag is set.
     */
    bool getStorePressureRawFlag()         const {return mCommandLineParameters.getStorePressureRawFlag();};
    /**
     * @brief  Is --p_rms set?
     * @return True if the flag is set.
     */
    bool getStorePressureRmsFlag()         const {return mCommandLineParameters.getStorePressureRmsFlag();};
    /**
     * @brief  Is --p_max set?
     * @return True if the flag is set.
     */
    bool getStorePressureMaxFlag()         const {return mCommandLineParameters.getStorePressureMaxFlag();};
    /**
     * @brief  Is --p_min set?
     * @return True if the flag is set.
     */
    bool getStorePressureMinFlag()         const {return mCommandLineParameters.getStorePressureMinFlag();};
    /**
     * @brief  Is --p_max_all set?
     * @return True if the flag is set.
     */
    bool getStorePressureMaxAllFlag()      const {return mCommandLineParameters.getStorePressureMaxAllFlag();};
    /**
     * @brief  Is --p_min_all set?
     * @return true if the flag is set.
     */
    bool getStorePressureMinAllFlag()      const {return mCommandLineParameters.getStorePressureMinAllFlag();};
    /**
     * @brief  Is --p_final set?
     * @return True if the flag is set.
     */
    bool getStorePressureFinalAllFlag()    const {return mCommandLineParameters.getStorePressureFinalAllFlag();};


    /**
     * @brief  Is -u or --u_raw specified at the command line?
     * @return True if the flag is set.
     */
    bool getStoreVelocityRawFlag()         const {return mCommandLineParameters.getStoreVelocityRawFlag();};
    /**
     * @brief  Is --u_non_staggered_raw set?
     * @return True if the flag is set.
     */
    bool getStoreVelocityNonStaggeredRawFlag () const
    {
      return mCommandLineParameters.getStoreVelocityNonStaggeredRawFlag();
    };
    /**
     * @brief  Is --u_rms set?
     * @return True if the flag is set.
     */
    bool getStoreVelocityRmsFlag()         const {return mCommandLineParameters.getStoreVelocityRmsFlag();};
    /**
     * @brief  Is --u_max set?
     * @return True if the flag is set.
     */
    bool getVelocityMaxFlag()              const {return mCommandLineParameters.getStoreVelocityMaxFlag();};
    /**
     * @brief  Is --u_min set?
     * @return True if the flag is set.
     */
    bool getStoreVelocityMinFlag()         const {return mCommandLineParameters.getStoreVelocityMinFlag();};
    /**
     * @brief  Is --u_max_all set?
     * @return True if the flag is set.
     */
    bool getStoreVelocityMaxAllFlag()      const {return mCommandLineParameters.getStoreVelocityMaxAllFlag();};
    /**
     * @brief  Is --u_min set?
     * @return True if the flag is set.
     */
    bool getStoreStoreVelocityMinAllFlag() const {return mCommandLineParameters.getStoreVelocityMinAllFlag();};
    /**
     * @brief  Is --u_final set?
     * @return True if the flag is set.
     */
    bool getStoreVelocityFinalAllFlag()    const {return mCommandLineParameters.getStoreVelocityFinalAllFlag();};

    /**
     * @brief  Is --copy_mask set set?
     * @return True if the flag is set.
     */
    bool getCopySensorMaskFlag()           const {return mCommandLineParameters.getCopySensorMaskFlag();};

  protected:
    /// Constructor.
    Parameters();

  private:
    /// Class with CUDA Parameters (runtime setup)
    CudaParameters        mCudaParameters;
    /// Class with command line parameters
    CommandLineParameters mCommandLineParameters;

    /// Handle to the input HDF5 file.
    Hdf5File mInputFile;
    /// Handle to the output HDF5 file.
    Hdf5File mOutputFile;
    /// Handle to the checkpoint HDF5 file.
    Hdf5File mCheckpointFile;

    /// Handle to the file header.
    Hdf5FileHeader mFileHeader;

    /// Full 3D dimension sizes.
    DimensionSizes mFullDimensionSizes;
    /// Reduced 3D dimension sizes.
    DimensionSizes mReducedDimensionSizes;

    /// Total number of time steps.
    size_t  mNt;
    /// Actual time index (time step of the simulation).
    size_t  mTimeIndex;

    /// Time step size.
    float mDt;
    /// Spatial displacement in x.
    float mDx;
    /// Spatial displacement in y.
    float mDy;
    /// Spatial displacement in z.
    float mDz;

    /// Reference sound speed.
    float mCRef;
    /// Is sound speed in the medium homogeneous?
    bool  mC0ScalarFlag;
    /// Scalar value of sound speed.
    float mC0Scalar;

    /// Is density in the medium homogeneous?
    bool  mRho0ScalarFlag;
    /// Homogeneous medium density.
    float mRho0Scalar;
    /// Homogeneous medium density on staggered grid in x direction.
    float mRho0SgxScalar;
    ///  Homogeneous medium density on staggered grid in y direction.
    float mRho0SgyScalar;
    ///  Homogeneous medium density on staggered grid in z direction.
    float mRho0SgzScalar;


    /// Enable non uniform grid?
    size_t mNonUniformGridFlag;
    /// Is the simulation absrobing or lossless?
    size_t mAbsorbingFlag;
    /// Is the wave propagation nonlinear?
    size_t mNonLinearFlag;

    /// Is alpha absorption coefficient homogeneous?
    bool  mAlphaCoeffScalarFlag;
    /// Alpha absorption coefficient.
    float mAlphaCoeffScalar;
    /// Alpha power value for the absorption law.
    float mAlphaPower;

    /// Absorb eta coefficient for homogeneous medium.
    float mAbsorbEtaScalar;
    /// Absorb tau coefficient for homogeneous medium.
    float mAbsorbTauScalar;

    /// Is nonlinear coefficient homogeneous in the medium?
    bool  mBOnAScalarFlag;
    /// Nonlinear coefficient for homogenous medium.
    float mBOnAScalar;

    /// Depth of the perfectly matched layer in x.
    size_t mPmlXSize;
    /// Depth of the perfectly matched layer in y.
    size_t mPmlYSize;
    /// Depth of the perfectly matched layer in z.
    size_t mPmlZSize;

    /// Perfectly matched layer attenuation in x.
    float mPmlXAlpha;
    /// Perfectly matched layer attenuation in y.
    float mPmlYAlpha;
    /// Perfectly matched layer attenuation in z.
    float mPmlZAlpha;

    /// Pressure source flag.
    size_t mPressureSourceFlag;
    /// Initial pressure source flag (p0).
    size_t mInitialPressureSourceFlag;
    /// Transducer source flag.
    size_t mTransducerSourceFlag;

    /// Velocity in x source flag.
    size_t mVelocityXSourceFlag;
    /// Velocity in y source flag.
    size_t mVelocityYSourceFlag;
    /// Velocity in z source flag.
    size_t mVelocityZSourceFlag;

    /// Spatial size of the pressure source.
    size_t mPressureSourceIndexSize;
    /// Spatial size of the transducer source.
    size_t mTransducerSourceInputSize;
    /// Spatial size of the velocity source.
    size_t mVelocitySourceIndexSize;

    /// Pressure source mode.
    size_t mPressureSourceMode;
    /// Number of time series in the pressure source.
    size_t mPressureSourceMany;

    /// Velocity source mode.
    size_t mVelocitySourceMany;
    /// Number of time series in the velocity sources.
    size_t mVelocitySourceMode;

    /// Sensor mask type (index / corners).
    SensorMaskType mSensorMaskType;
    /// How many elements there are in the linear mask
    size_t mSensorMaskIndexSize;
    /// Sensor_mask_corners_size - how many cuboids are in the mask.
    size_t mSensorMaskCornersSize;

    /// Singleton flag
    static bool        sParametersInstanceFlag;
    /// Singleton instance
    static Parameters* sPrametersInstance;
};// end of Parameters
//----------------------------------------------------------------------------------------------------------------------

#endif	/* ParametersH */
