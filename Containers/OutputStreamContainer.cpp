/**
 * @file      OutputStreamContainer.cpp
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The implementation file for the output stream container.
 *
 * @version   kspaceFirstOrder 3.6
 *
 * @date      04 December  2014, 11:41 (created) \n
 *            06 March     2019, 12:57 (revised)
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

#include <Parameters/Parameters.h>
#include <Containers/OutputStreamContainer.h>

#include <OutputStreams/BaseOutputStream.h>
#include <OutputStreams/IndexOutputStream.h>
#include <OutputStreams/CuboidOutputStream.h>
#include <OutputStreams/WholeDomainOutputStream.h>

//--------------------------------------------------------------------------------------------------------------------//
//---------------------------------------------------- Constants -----------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

//--------------------------------------------------------------------------------------------------------------------//
//-------------------------------------------------- Public methods --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Default constructor.
 */
OutputStreamContainer::OutputStreamContainer()
  : mContainer() {

} // end of OutputStreamContainer
//----------------------------------------------------------------------------------------------------------------------

/**
 * Destructor.
 */
OutputStreamContainer::~OutputStreamContainer() {
  mContainer.clear();
} // end of ~OutputStreamContainer
//----------------------------------------------------------------------------------------------------------------------

/**
 * Add all streams in the simulation to the container, set all streams records here!
 */
void OutputStreamContainer::init(MatrixContainer& matrixContainer) {
  Parameters& params = Parameters::getInstance();

  // shortcuts for long data types
  using OI = OutputStreamIdx;
  using MI = MatrixContainer::MatrixIdx;
  using RO = BaseOutputStream::ReduceOperator;

  const bool is3DSimulation = params.isSimulation3D();

  if (!params.getOnlyPostProcessingFlag()) {
    //--------------------------------------------------- pressure -----------------------------------------------------//
    if (params.getStorePressureRawFlag()) {
      mContainer[OI::kPressureRaw] = createOutputStream(matrixContainer, MI::kP, kPressureRawName, RO::kNone);
    }

    if (params.getStorePressureCFlag()) {
      mContainer[OI::kPressureC] = createOutputStream(matrixContainer, MI::kP, kPressureRawName + kCompressSuffix, RO::kC);
    }

    if (params.getStorePressureRmsFlag()) {
      mContainer[OI::kPressureRms] = createOutputStream(matrixContainer, MI::kP, kPressureRmsName, RO::kRms);
    }

    if (params.getStorePressureMaxFlag()) {
      mContainer[OI::kPressureMax] = createOutputStream(matrixContainer, MI::kP, kPressureMaxName, RO::kMax);
    }

    if (params.getStorePressureMinFlag()) {
      mContainer[OI::kPressureMin] = createOutputStream(matrixContainer, MI::kP, kPressureMinName, RO::kMin);
    }

    if (params.getStorePressureMaxAllFlag()) {
      mContainer[OI::kPressureMaxAll] = new WholeDomainOutputStream(params.getOutputFile(),
                                                                    kPressureMaxAllName,
                                                                    matrixContainer.getMatrix<RealMatrix>(MI::kP),
                                                                    RO::kMax);
    }

    if (params.getStorePressureMinAllFlag()) {
      mContainer[OI::kPressureMinAll] = new WholeDomainOutputStream(params.getOutputFile(),
                                                                    kPressureMinAllName,
                                                                    matrixContainer.getMatrix<RealMatrix>(MI::kP),
                                                                    RO::kMin);
    }

    //-------------------------------------------------- velocity ------------------------------------------------------//
    if (params.getStoreVelocityRawFlag()) {
      mContainer[OI::kVelocityXRaw] = createOutputStream(matrixContainer, MI::kUxSgx, kUxName, RO::kNone);
      mContainer[OI::kVelocityYRaw] = createOutputStream(matrixContainer, MI::kUySgy, kUyName, RO::kNone);
      if (is3DSimulation) {
        mContainer[OI::kVelocityZRaw] = createOutputStream(matrixContainer, MI::kUzSgz, kUzName, RO::kNone);
      }
    }

    if (params.getStoreVelocityCFlag()) {
      mContainer[OI::kVelocityXC] = createOutputStream(matrixContainer, MI::kUxSgx, kUxName + kCompressSuffix, RO::kC);
      mContainer[OI::kVelocityYC] = createOutputStream(matrixContainer, MI::kUySgy, kUyName + kCompressSuffix, RO::kC);
      if (is3DSimulation) {
        mContainer[OI::kVelocityZC] = createOutputStream(matrixContainer, MI::kUzSgz, kUzName + kCompressSuffix, RO::kC);
      }
    }

    if (params.getStoreVelocityNonStaggeredRawFlag()) {
      mContainer[OI::kVelocityXNonStaggeredRaw] = createOutputStream(matrixContainer,
                                                                     MI::kUxShifted,
                                                                     kUxNonStaggeredName,
                                                                     RO::kNone);
      mContainer[OI::kVelocityYNonStaggeredRaw] = createOutputStream(matrixContainer,
                                                                     MI::kUyShifted,
                                                                     kUyNonStaggeredName,
                                                                     RO::kNone);
      if (is3DSimulation) {
        mContainer[OI::kVelocityZNonStaggeredRaw] = createOutputStream(matrixContainer,
                                                                       MI::kUzShifted,
                                                                       kUzNonStaggeredName,
                                                                       RO::kNone);
      }
    }

    if (params.getStoreVelocityNonStaggeredCFlag()) {
      mContainer[OI::kVelocityXNonStaggeredC] = createOutputStream(matrixContainer, MI::kUxShifted, kUxNonStaggeredName + kCompressSuffix, RO::kC);
      mContainer[OI::kVelocityYNonStaggeredC] = createOutputStream(matrixContainer, MI::kUyShifted, kUyNonStaggeredName + kCompressSuffix, RO::kC);
      if (is3DSimulation) {
        mContainer[OI::kVelocityZNonStaggeredC] = createOutputStream(matrixContainer, MI::kUzShifted, kUzNonStaggeredName + kCompressSuffix, RO::kC);
      }
    }

    if (params.getStoreVelocityRmsFlag()) {
      mContainer[OI::kVelocityXRms] = createOutputStream(matrixContainer, MI::kUxSgx, kUxRmsName, RO::kRms);
      mContainer[OI::kVelocityYRms] = createOutputStream(matrixContainer, MI::kUySgy, kUyRmsName, RO::kRms);
      if (is3DSimulation) {
        mContainer[OI::kVelocityZRms] = createOutputStream(matrixContainer, MI::kUzSgz, kUzRmsName, RO::kRms);
      }
    }

    if (params.getStoreVelocityMaxFlag()) {
      mContainer[OI::kVelocityXMax] = createOutputStream(matrixContainer, MI::kUxSgx, kUxMaxName, RO::kMax);
      mContainer[OI::kVelocityYMax] = createOutputStream(matrixContainer, MI::kUySgy, kUyMaxName, RO::kMax);
      if (is3DSimulation) {
        mContainer[OI::kVelocityZMax] = createOutputStream(matrixContainer, MI::kUzSgz, kUzMaxName, RO::kMax);
      }
    }

    if (params.getStoreVelocityMinFlag()) {
      mContainer[OI::kVelocityXMin] = createOutputStream(matrixContainer, MI::kUxSgx, kUxMinName, RO::kMin);
      mContainer[OI::kVelocityYMin] = createOutputStream(matrixContainer, MI::kUySgy, kUyMinName, RO::kMin);
      if (is3DSimulation) {
        mContainer[OI::kVelocityZMin] = createOutputStream(matrixContainer, MI::kUzSgz, kUzMinName, RO::kMin);
      }
    }

    if (params.getStoreVelocityMaxAllFlag()) {
      mContainer[OI::kVelocityXMaxAll] = new WholeDomainOutputStream(params.getOutputFile(),
                                                                     kUxMaxAllName,
                                                                     matrixContainer.getMatrix<RealMatrix>(MI::kUxSgx),
                                                                     RO::kMax);
      mContainer[OI::kVelocityYMaxAll] = new WholeDomainOutputStream(params.getOutputFile(),
                                                                     kUyMaxAllName,
                                                                     matrixContainer.getMatrix<RealMatrix>(MI::kUySgy),
                                                                     RO::kMax);
      if (is3DSimulation) {
        mContainer[OI::kVelocityZMaxAll] = new WholeDomainOutputStream(params.getOutputFile(),
                                                                       kUzMaxAllName,
                                                                       matrixContainer.getMatrix<RealMatrix>(MI::kUzSgz),
                                                                       RO::kMax);
      }
    }

    if (params.getStoreVelocityMinAllFlag()) {
      mContainer[OI::kVelocityXMinAll] = new WholeDomainOutputStream(params.getOutputFile(),
                                                                     kUxMinAllName,
                                                                     matrixContainer.getMatrix<RealMatrix>(MI::kUxSgx),
                                                                     RO::kMin);
      mContainer[OI::kVelocityYMinAll] = new WholeDomainOutputStream(params.getOutputFile(),
                                                                     kUyMinAllName,
                                                                     matrixContainer.getMatrix<RealMatrix>(MI::kUySgy),
                                                                     RO::kMin);
      if (is3DSimulation) {
        mContainer[OI::kVelocityZMinAll] = new WholeDomainOutputStream(params.getOutputFile(),
                                                                       kUzMinAllName,
                                                                       matrixContainer.getMatrix<RealMatrix>(MI::kUzSgz),
                                                                       RO::kMin);
      }
    }
  }

  //------------------------------------------ average intensity, Q term ---------------------------------------------//
  if (params.getStoreQTermFlag() || params.getStoreIntensityAvgFlag()) {
    if (!params.getStorePressureRawFlag() && !params.getOnlyPostProcessingFlag()) {
      mContainer[OI::kPressureRaw] = createOutputStream(matrixContainer, MI::kP, kPressureRawName, RO::kNone);
    }

    if (!params.getStoreVelocityNonStaggeredRawFlag() && !params.getOnlyPostProcessingFlag()) {
      mContainer[OI::kVelocityXNonStaggeredRaw] = createOutputStream(matrixContainer, MI::kUxShifted, kUxNonStaggeredName, RO::kNone);
      mContainer[OI::kVelocityYNonStaggeredRaw] = createOutputStream(matrixContainer, MI::kUyShifted, kUyNonStaggeredName, RO::kNone);
      if (is3DSimulation) {
        mContainer[OI::kVelocityZNonStaggeredRaw] = createOutputStream(matrixContainer, MI::kUzShifted, kUzNonStaggeredName, RO::kNone);
      }
    }

    if (!params.getStoreIntensityAvgFlag()) {
      mContainer[OI::kIntensityXAvg] = createOutputStream(matrixContainer, MI::kP, kIxAvgName, RO::kIAvg, true);
      mContainer[OI::kIntensityYAvg] = createOutputStream(matrixContainer, MI::kP, kIyAvgName, RO::kIAvg, true);
      if (is3DSimulation) {
        mContainer[OI::kIntensityZAvg] = createOutputStream(matrixContainer, MI::kP, kIzAvgName, RO::kIAvg, true);
      }
    } else {
      mContainer[OI::kIntensityXAvg] = createOutputStream(matrixContainer, MI::kP, kIxAvgName, RO::kIAvg);
      mContainer[OI::kIntensityYAvg] = createOutputStream(matrixContainer, MI::kP, kIyAvgName, RO::kIAvg);
      if (is3DSimulation) {
        mContainer[OI::kIntensityZAvg] = createOutputStream(matrixContainer, MI::kP, kIzAvgName, RO::kIAvg);
      }
    }

    if (params.getStoreQTermFlag()) {
      mContainer[OI::kQTerm] = createOutputStream(matrixContainer, MI::kP, kQTermName, RO::kQTerm);
    }
  }

  if (params.getStoreQTermCFlag() || params.getStoreIntensityAvgCFlag()) {
    if (!params.getStorePressureCFlag() && !params.getOnlyPostProcessingFlag()) {
      mContainer[OI::kPressureC] = createOutputStream(matrixContainer, MI::kP, kPressureRawName + kCompressSuffix, RO::kC, true);
    }

    if (!params.getStoreVelocityNonStaggeredCFlag() && !params.getOnlyPostProcessingFlag()) {
      mContainer[OI::kVelocityXNonStaggeredC] = createOutputStream(matrixContainer, MI::kUxShifted, kUxNonStaggeredName + kCompressSuffix, RO::kC, true);
      mContainer[OI::kVelocityYNonStaggeredC] = createOutputStream(matrixContainer, MI::kUyShifted, kUyNonStaggeredName + kCompressSuffix, RO::kC, true);
      if (is3DSimulation) {
        mContainer[OI::kVelocityZNonStaggeredC] = createOutputStream(matrixContainer, MI::kUzShifted, kUzNonStaggeredName + kCompressSuffix, RO::kC, true);
      }
    }

    if (!params.getStoreIntensityAvgCFlag()) {
      mContainer[OI::kIntensityXAvgC] = createOutputStream(matrixContainer, MI::kP, kIxAvgName + kCompressSuffix, RO::kIAvgC, true);
      mContainer[OI::kIntensityYAvgC] = createOutputStream(matrixContainer, MI::kP, kIyAvgName + kCompressSuffix, RO::kIAvgC, true);
      if (is3DSimulation) {
        mContainer[OI::kIntensityZAvgC] = createOutputStream(matrixContainer, MI::kP, kIzAvgName + kCompressSuffix, RO::kIAvgC, true);
      }
    } else {
      mContainer[OI::kIntensityXAvgC] = createOutputStream(matrixContainer, MI::kP, kIxAvgName + kCompressSuffix, RO::kIAvgC);
      mContainer[OI::kIntensityYAvgC] = createOutputStream(matrixContainer, MI::kP, kIyAvgName + kCompressSuffix, RO::kIAvgC);
      if (is3DSimulation) {
        mContainer[OI::kIntensityZAvgC] = createOutputStream(matrixContainer, MI::kP, kIzAvgName + kCompressSuffix, RO::kIAvgC);
      }
    }

    if (params.getStoreQTermCFlag()) {
      mContainer[OI::kQTermC] = createOutputStream(matrixContainer, MI::kP, kQTermName + kCompressSuffix, RO::kQTermC);
    }
  }

} // end of init
//----------------------------------------------------------------------------------------------------------------------

/**
 * Create all streams.
 */
void OutputStreamContainer::createStreams() {
  for (const auto& it : mContainer) {
    if (it.second) {
      it.second->create();
    }
  }
} // end of createStreams
//----------------------------------------------------------------------------------------------------------------------

/**
 * Reopen all streams after restarting from checkpoint.
 */
void OutputStreamContainer::reopenStreams() {
  for (const auto& it : mContainer) {
    if (it.second) {
      it.second->reopen();
    }
  }
} // end of reopenStreams
//----------------------------------------------------------------------------------------------------------------------

/**
 * Sample all streams.
 */
void OutputStreamContainer::sampleStreams() {
  for (const auto& it : mContainer) {
    if (it.second) {
      it.second->sample();
    }
  }
} // end of sampleStreams
//----------------------------------------------------------------------------------------------------------------------

/**
 * Flush stream data to disk.
 */
void OutputStreamContainer::flushRawStreams() {
  for (const auto& it : mContainer) {
    if (it.second) {
      it.second->flushRaw();
    }
  }
  for (const auto& it : mContainer) {
    if (it.second) {
      it.second->postSample();
    }
  }
  for (const auto& it : mContainer) {
    if (it.second) {
      it.second->postSample2();
    }
  }
} // end of sampleStreams
//----------------------------------------------------------------------------------------------------------------------

/**
 * Checkpoint streams without post-processing (flush to the file).
 */
void OutputStreamContainer::checkpointStreams() {
  for (const auto& it : mContainer) {
    if (it.second) {
      it.second->checkpoint();
    }
  }
} // end of checkpointStreams
//----------------------------------------------------------------------------------------------------------------------

/**
 * Post-process all streams and flush them to the file.
 */
void OutputStreamContainer::postProcessStreams() {
  for (const auto& it : mContainer) {
    if (it.second) {
      it.second->postProcess();
    }
  }
} // end of postProcessStreams
//----------------------------------------------------------------------------------------------------------------------

/**
 * Post-process all streams and flush them to the file.
 */
void OutputStreamContainer::postProcessStreams2() {
  for (const auto& it : mContainer) {
    if (it.second) {
      it.second->postProcess2();
    }
  }
} // end of postProcessStreams
//----------------------------------------------------------------------------------------------------------------------

/**
 * Close all streams (apply post-processing if necessary, flush data and close).
 */
void OutputStreamContainer::closeStreams() {
  for (const auto& it : mContainer) {
    if (it.second) {
      it.second->close();
    }
  }
} // end of closeStreams
//----------------------------------------------------------------------------------------------------------------------

/**
 *  Free all streams - destroy them.
 */
void OutputStreamContainer::freeStreams() {
  for (auto& it : mContainer) {
    if (it.second) {
      delete it.second;
      it.second = nullptr;
    }
  }
  mContainer.clear();
} // end of freeStreams
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------ Protected methods -------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Create a new output stream.
 */
BaseOutputStream* OutputStreamContainer::createOutputStream(MatrixContainer& matrixContainer,
                                                            const MatrixContainer::MatrixIdx sampledMatrixIdx,
                                                            const MatrixName& fileObjectName,
                                                            const BaseOutputStream::ReduceOperator reduceOp,
                                                            bool doNotSaveFlag) {
  using MatrixIdx = MatrixContainer::MatrixIdx;
  Parameters& params = Parameters::getInstance();
  BaseOutputStream* stream = nullptr;

  if (params.getSensorMaskType() == Parameters::SensorMaskType::kIndex) {
    stream = new IndexOutputStream(params.getOutputFile(),
                                   fileObjectName,
                                   matrixContainer.getMatrix<RealMatrix>(sampledMatrixIdx),
                                   matrixContainer.getMatrix<IndexMatrix>(MatrixIdx::kSensorMaskIndex),
                                   reduceOp,
                                   this,
                                   doNotSaveFlag);
  } else {
    stream = new CuboidOutputStream(params.getOutputFile(),
                                    fileObjectName,
                                    matrixContainer.getMatrix<RealMatrix>(sampledMatrixIdx),
                                    matrixContainer.getMatrix<IndexMatrix>(MatrixIdx::kSensorMaskCorners),
                                    reduceOp,
                                    this,
                                    doNotSaveFlag);
  }
  return stream;
} // end of createOutputStream
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Private methods --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//
