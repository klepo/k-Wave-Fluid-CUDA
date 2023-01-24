/**
 * @file      CommandLineParameters.cpp
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The implementation file containing the command line parameters.
 *
 * @version   kspaceFirstOrder 3.6
 *
 * @date      29 August    2012, 11:25 (created) \n
 *            12 March     2019, 10:59 (revised)
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

//Linux build
#ifdef __linux__
#include <getopt.h>
#endif

//Windows build
#ifdef _WIN64
#include <GetoptWin64/Getopt.h>
#endif

#include <cstring>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <stdexcept>

#include <Parameters/CudaParameters.h>
#include <Parameters/CommandLineParameters.h>
#include <Logger/Logger.h>

using std::string;

//--------------------------------------------------------------------------------------------------------------------//
//---------------------------------------------------- Constants -----------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Print usage.
 */
void CommandLineParameters::printUsage() {
  Logger::log(Logger::LogLevel::kBasic, kOutFmtUsagePart1);

#ifdef _OPENMP
  Logger::log(Logger::LogLevel::kBasic, kOutFmtUsageThreads, omp_get_num_procs());
#endif

  Logger::log(Logger::LogLevel::kBasic, kOutFmtUsagePart2, kDefaultProgressPrintInterval, kDefaultCompressionLevel);
} // end of printUsage
//----------------------------------------------------------------------------------------------------------------------

/**
 * Print out commandline parameters.
 */
void CommandLineParameters::printComandlineParamers() {
  Logger::log(Logger::LogLevel::kAdvanced, kOutFmtSeparator);

  Logger::log(Logger::LogLevel::kAdvanced,
              Logger::wordWrapString(kOutFmtInputFile + mInputFileName, kErrFmtPathDelimiters, 15).c_str());

  Logger::log(Logger::LogLevel::kAdvanced,
              Logger::wordWrapString(kOutFmtOutputFile + mOutputFileName, kErrFmtPathDelimiters, 15).c_str());

  if (isCheckpointEnabled()) {
    Logger::log(Logger::LogLevel::kAdvanced,
                Logger::wordWrapString(kOutFmtCheckpointFile + mCheckpointFileName, kErrFmtPathDelimiters, 15).c_str());

    Logger::log(Logger::LogLevel::kAdvanced, kOutFmtSeparator);
    if (mCheckpointInterval > 0) {
      Logger::log(Logger::LogLevel::kAdvanced, kOutFmtCheckpointInterval, mCheckpointInterval);
    }

    if (mCheckpointTimeSteps > 0) {
      Logger::log(Logger::LogLevel::kAdvanced, kOutFmtCheckpointTimeSteps, mCheckpointTimeSteps);
    }
  } else {
    Logger::log(Logger::LogLevel::kAdvanced, kOutFmtSeparator);
  }

  Logger::log(Logger::LogLevel::kAdvanced, kOutFmtCompressionLevel, mCompressionLevel);

  Logger::log(Logger::LogLevel::kFull, kOutFmtPrintProgressIntrerval, mProgressPrintInterval);

  if ((mStoreIntensityAvgFlag || mStoreQTermFlag) && mBlockSize > 0) {
    Logger::log(Logger::LogLevel::kFull, kOutFmtBlockSize, mBlockSize);
  }

  if (mBenchmarkFlag) {
    Logger::log(Logger::LogLevel::kFull, kOutFmtBenchmarkTimeStep, mBenchmarkTimeStepCount);
  }

  if (mStorePressureCFlag || mStoreVelocityCFlag || mStoreVelocityNonStaggeredCFlag || mStoreIntensityAvgCFlag || mStoreQTermCFlag) {
    Logger::log(Logger::LogLevel::kAdvanced, kOutFmtSeparator);
    Logger::log(Logger::LogLevel::kBasic, kOutFmtCompressionSettings, mFrequency, mPeriod, mMOS, mHarmonics);
    Logger::log(Logger::LogLevel::kAdvanced, kOutFmtSeparator);
  }

  Logger::log(Logger::LogLevel::kAdvanced, kOutFmtSamplingFlags);

  string sampledQuantitiesList = "";
  // Sampled p quantities

  if (mStorePressureRawFlag) {
    sampledQuantitiesList += "p_raw, ";
  }
  if (mStorePressureCFlag) {
    sampledQuantitiesList += "p_c, ";
  }
  if (mStorePressureRmsFlag) {
    sampledQuantitiesList += "p_rms, ";
  }
  if (mStorePressureMaxFlag) {
    sampledQuantitiesList += "p_max, ";
  }
  if (mStorePressureMinFlag) {
    sampledQuantitiesList += "p_min, ";
  }
  if (mStorePressureMaxAllFlag) {
    sampledQuantitiesList += "p_max_all, ";
  }
  if (mStorePressureMinAllFlag) {
    sampledQuantitiesList += "p_min_all, ";
  }
  if (mStorePressureFinalAllFlag) {
    sampledQuantitiesList += "p_final, ";
  }

  // Sampled u quantities
  if (mStoreVelocityRawFlag) {
    sampledQuantitiesList += "u_raw, ";
  }
  if (mStoreVelocityCFlag) {
    sampledQuantitiesList += "u_c, ";
  }
  if (mStoreVelocityRmsFlag) {
    sampledQuantitiesList += "u_rms, ";
  }
  if (mStoreVelocityMaxFlag) {
    sampledQuantitiesList += "u_max, ";
  }
  if (mStoreVelocityMinFlag) {
    sampledQuantitiesList += "u_min, ";
  }
  if (mStoreVelocityMaxAllFlag) {
    sampledQuantitiesList += "u_max_all, ";
  }
  if (mStoreVelocityMinAllFlag) {
    sampledQuantitiesList += "u_min_all, ";
  }
  if (mStoreVelocityFinalAllFlag) {
    sampledQuantitiesList += "u_final, ";
  }
  if (mStoreVelocityNonStaggeredRawFlag) {
    sampledQuantitiesList += "u_non_staggered_raw, ";
  }
  if (mStoreVelocityNonStaggeredCFlag) {
    sampledQuantitiesList += "u_non_staggered_c, ";
  }
  if (mStoreIntensityAvgFlag) {
    sampledQuantitiesList += "I_avg, ";
  }
  if (mStoreIntensityAvgCFlag) {
    sampledQuantitiesList += "I_avg_c, ";
  }
  if (mStoreQTermFlag) {
    sampledQuantitiesList += "Q_term, ";
  }
  if (mStoreQTermCFlag) {
    sampledQuantitiesList += "Q_term_c, ";
  }

  // remove comma and space symbols
  if (sampledQuantitiesList.length() > 0) {
    sampledQuantitiesList.pop_back();
    sampledQuantitiesList.pop_back();
  }

  Logger::log(Logger::LogLevel::kAdvanced,
              Logger::wordWrapString(sampledQuantitiesList, " ", 2).c_str());

  Logger::log(Logger::LogLevel::kAdvanced, kOutFmtSeparator);

  Logger::log(Logger::LogLevel::kAdvanced, kOutFmtSamplingStartsAt, mSamplingStartTimeStep + 1);

  if (mCopySensorMaskFlag) {
    Logger::log(Logger::LogLevel::kAdvanced, kOutFmtCopySensorMask);
  }
} // end of printComandlineParamers
//----------------------------------------------------------------------------------------------------------------------

/**
 * Parse command line.
 */
void CommandLineParameters::parseCommandLine(int argc, char** argv) {
  char c;
  int longIndex = -1;
  bool checkpointFlag = false;

  constexpr int errorLineIndent = 9;

// all optional arguments are in fact requested. This was chosen to prevent
// getopt error messages and provide custom error handling.
#ifdef _OPENMP
  const char* shortOpts = "i:o:r:c:t:g:puhs:";
#else
  const char* shortOpts = "i:o:r:c:g:puhs:";
#endif

  const struct option longOpts[] =
    {
      {"benchmark", required_argument, nullptr, 1},
      {"copy_sensor_mask", no_argument, nullptr, 2},
      {"checkpoint_file", required_argument, nullptr, 3},
      {"checkpoint_interval", required_argument, nullptr, 4},
      {"checkpoint_timesteps", required_argument, nullptr, 5},
      {"help", no_argument, nullptr, 'h'},
      {"verbose", required_argument, nullptr, 6},
      {"version", no_argument, nullptr, 7},

      {"p_raw", no_argument, nullptr, 'p'},
      {"p_c", no_argument, nullptr, 9},
      {"p_rms", no_argument, nullptr, 10},
      {"p_max", no_argument, nullptr, 11},
      {"p_min", no_argument, nullptr, 12},
      {"p_max_all", no_argument, nullptr, 13},
      {"p_min_all", no_argument, nullptr, 14},
      {"p_final", no_argument, nullptr, 15},

      {"frequency", required_argument, nullptr, 16},
      {"period", required_argument, nullptr, 17},
      {"mos", required_argument, nullptr, 18},
      {"harmonics", required_argument, nullptr, 19},

      {"u_raw", no_argument, nullptr, 'u'},
      {"u_rms", no_argument, nullptr, 20},
      {"u_max", no_argument, nullptr, 21},
      {"u_min", no_argument, nullptr, 22},
      {"u_max_all", no_argument, nullptr, 23},
      {"u_min_all", no_argument, nullptr, 24},
      {"u_final", no_argument, nullptr, 25},
      {"u_non_staggered_raw", no_argument, nullptr, 26},
      {"u_c", no_argument, nullptr, 27},
      {"u_non_staggered_c", no_argument, nullptr, 28},
      {"I_avg", no_argument, nullptr, 29},
      {"I_avg_c", no_argument, nullptr, 30},
      {"Q_term", no_argument, nullptr, 31},
      {"Q_term_c", no_argument, nullptr, 32},

      {"post", no_argument, nullptr, 33},
      {"block_size", required_argument, nullptr, 34},
      {"no_overlap", no_argument, nullptr, 35},
      {"40-bit_complex", no_argument, nullptr, 36},

      {nullptr, no_argument, nullptr, 0}};

  // all optional arguments are in fact requested. This was chosen to prevent
  // getopt error messages and provide custom error handling.
  opterr = 0;

  // Short parameters //
  while ((c = getopt_long(argc, argv, shortOpts, longOpts, &longIndex)) != -1) {
    switch (c) {
      case 'i': {
        // test if the file was correctly entered (if not, getopt could eat
        // the following parameter)
        if ((optarg != nullptr) && ((strlen(optarg) > 0) && (optarg[0] != '-'))) {
          mInputFileName = optarg;
        } else {
          printUsage();
          Logger::errorAndTerminate(Logger::wordWrapString(kErrFmtNoInputFile, " ", errorLineIndent));
        }
        break;
      }

      case 'o': {
        // test if the wile was correctly entered (if not, getopt could eat
        // the following parameter)
        if ((optarg != nullptr) && ((strlen(optarg) > 0) && (optarg[0] != '-'))) {
          mOutputFileName = optarg;
        } else {
          printUsage();
          Logger::errorAndTerminate(Logger::wordWrapString(kErrFmtNoOutputFile, " ", errorLineIndent));
        }
        break;
      }

      case 'r': {
        try {
          float convertedValue = std::stof(optarg);
          if ((convertedValue <= 0.0f) || (convertedValue >= 100.0f)) {
            throw std::invalid_argument("-r");
          }
          mProgressPrintInterval = std::stof(optarg);
        } catch (...) {
          printUsage();
          Logger::errorAndTerminate(Logger::wordWrapString(kErrFmtNoProgressPrintInterval, " ", errorLineIndent));
        }
        break;
      }

#ifdef _OPENMP
      case 't': {
        try {
          if (std::stoi(optarg) < 1) {
            throw std::invalid_argument("-t");
          }
          mNumberOfThreads = std::stoll(optarg);
        } catch (...) {
          printUsage();
          Logger::errorAndTerminate(Logger::wordWrapString(kErrFmtInvalidNumberOfThreads, " ", errorLineIndent));
        }
        break;
      }
#endif
      case 'g': {
        try {
          mCudaDeviceIdx = std::stoi(optarg);
          if (mCudaDeviceIdx < 0) {
            throw std::invalid_argument("-g");
          }
        } catch (...) {
          printUsage();
          Logger::errorAndTerminate(Logger::wordWrapString(kErrFmtNoDeviceIndex, " ", errorLineIndent));
        }
        break;
      }

      case 'c': {
        try {
          int covertedValue = std::stoi(optarg);
          if ((covertedValue < 0) || (covertedValue > 9)) {
            throw std::invalid_argument("-c");
          }
          mCompressionLevel = std::stoll(optarg);
        } catch (...) {
          printUsage();
          Logger::errorAndTerminate(Logger::wordWrapString(kErrFmtNoCompressionLevel, " ", errorLineIndent));
        }
        break;
      }

      case 'h': {
        printUsage();
        exit(EXIT_SUCCESS);
      }

      case 's': {
        try {
          if (std::stoll(optarg) < 1) {
            throw std::invalid_argument("-s");
          }
          mSamplingStartTimeStep = std::stoll(optarg) - 1;
        } catch (...) {
          printUsage();
          Logger::errorAndTerminate(Logger::wordWrapString(kErrFmtNoSamplingStartTimeStep, " ", errorLineIndent));
        }
        break;
      }

      case 1: // benchmark
      {
        try {
          mBenchmarkFlag = true;
          if (std::stoll(optarg) <= 0) {
            throw std::invalid_argument("benchmark");
          }
          mBenchmarkTimeStepCount = std::stoll(optarg);
        } catch (...) {
          printUsage();
          Logger::errorAndTerminate(Logger::wordWrapString(kErrFmtNoBenchmarkTimeStep, " ", errorLineIndent));
        }
        break;
      }

      case 2: // copy_sensor_mask
      {
        mCopySensorMaskFlag = true;
        break;
      }

      case 3: // checkpoint_file
      {
        checkpointFlag = true;
        // test if the wile was correctly entered (if not, getopt could eat
        // the following parameter)
        if ((optarg != NULL) && ((strlen(optarg) > 0) && (optarg[0] != '-'))) {
          mCheckpointFileName = optarg;
        } else {
          printUsage();
          Logger::errorAndTerminate(Logger::wordWrapString(kErrFmtNoCheckpointFile, " ", errorLineIndent));
        }
        break;
      }

      case 4: // checkpoint_interval
      {
        try {
          checkpointFlag = true;
          if (std::stoll(optarg) <= 0) {
            throw std::invalid_argument("checkpoint_interval");
          }
          mCheckpointInterval = std::stoll(optarg);
        } catch (...) {
          printUsage();
          Logger::errorAndTerminate(Logger::wordWrapString(kErrFmtNoCheckpointInterval, " ", errorLineIndent));
        }
        break;
      }

      case 5: // checkpoint_timesteps
      {
        try {
          checkpointFlag = true;
          if (std::stoll(optarg) <= 0) {
            throw std::invalid_argument("checkpoint_timesteps");
          }
          mCheckpointTimeSteps = std::stoll(optarg);
        } catch (...) {
          printUsage();
          Logger::errorAndTerminate(Logger::wordWrapString(kErrFmtNoCheckpointTimeSteps, " ", errorLineIndent));
        }
        break;
      }

      case 6: // verbose
      {
        try {
          int verboseLevel = std::stoi(optarg);
          if ((verboseLevel < 0) || (verboseLevel > 2)) {
            throw std::invalid_argument("verbose");
          }
          Logger::setLevel(static_cast<Logger::LogLevel>(verboseLevel));
        } catch (...) {
          printUsage();
          Logger::errorAndTerminate(Logger::wordWrapString(kErrFmtNoVerboseLevel, " ", errorLineIndent));
        }
        break;
      }

      case 7: // version
      {
        mPrintVersionFlag = true;
        break;
      }

      case 'p': {
        mStorePressureRawFlag = true;
        break;
      }

      case 9: // p_c
      {
        mStorePressureCFlag = true;
        break;
      }

      case 10: // p_rms
      {
        mStorePressureRmsFlag = true;
        break;
      }

      case 11: // p_max
      {
        mStorePressureMaxFlag = true;
        break;
      }

      case 12: // p_min
      {
        mStorePressureMinFlag = true;
        break;
      }

      case 13: // p_max_all
      {
        mStorePressureMaxAllFlag = true;
        break;
      }

      case 14: // p_min_all
      {
        mStorePressureMinAllFlag = true;
        break;
      }

      case 15: // p_final
      {
        mStorePressureFinalAllFlag = true;
        break;
      }

      case 16: // frequency
      {
        try {
          if (std::stof(optarg) < 1) {
            throw std::invalid_argument("--frequency");
          }
          mFrequency = std::stof(optarg);
        } catch (...) {
          printUsage();
          Logger::errorAndTerminate(Logger::wordWrapString(kErrFmtInvalidFrequency, " ", errorLineIndent));
        }
        break;
      }

      case 17: // period
      {
        try {
          if (std::stof(optarg) < 1) {
            throw std::invalid_argument("--period");
          }
          mPeriod = std::stof(optarg);
        } catch (...) {
          printUsage();
          Logger::errorAndTerminate(Logger::wordWrapString(kErrFmtInvalidPeriod, " ", errorLineIndent));
        }
        break;
      }

      case 18: // mos
      {
        try {
          if (std::stoll(optarg) < 1) {
            throw std::invalid_argument("--mos");
          }
          mMOS = std::stoull(optarg);
        } catch (...) {
          printUsage();
          Logger::errorAndTerminate(Logger::wordWrapString(kErrFmtInvalidMOS, " ", errorLineIndent));
        }
        break;
      }

      case 19: // harmonics
      {
        try {
          if (std::stoll(optarg) < 1) {
            throw std::invalid_argument("--harmonics");
          }
          mHarmonics = std::stoull(optarg);
        } catch (...) {
          printUsage();
          Logger::errorAndTerminate(Logger::wordWrapString(kErrFmtInvalidHarmonics, " ", errorLineIndent));
        }
        break;
      }

      case 'u': {
        mStoreVelocityRawFlag = true;
        break;
      }

      case 27: // u_c
      {
        mStoreVelocityCFlag = true;
        break;
      }

      case 20: // u_rms
      {
        mStoreVelocityRmsFlag = true;
        break;
      }

      case 21: // u_max
      {
        mStoreVelocityMaxFlag = true;
        break;
      }

      case 22: // u_min
      {
        mStoreVelocityMinFlag = true;
        break;
      }

      case 23: // u_max_all
      {
        mStoreVelocityMaxAllFlag = true;
        break;
      }

      case 24: // u_min_all
      {
        mStoreVelocityMinAllFlag = true;
        break;
      }

      case 25: // u_final
      {
        mStoreVelocityFinalAllFlag = true;
        break;
      }

      case 26: // u_non_staggered_raw
      {
        mStoreVelocityNonStaggeredRawFlag = true;
        break;
      }

      case 28: // u_non_staggered_c
      {
        mStoreVelocityNonStaggeredCFlag = true;
        break;
      }

      case 29: // I_avg
      {
        mStoreIntensityAvgFlag = true;
        break;
      }

      case 30: // I_avg_c
      {
        mStoreIntensityAvgCFlag = true;
        break;
      }

      case 31: // Q_term
      {
        mStoreQTermFlag = true;
        break;
      }

      case 32: // Q_term_c
      {
        mStoreQTermCFlag = true;
        break;
      }

      case 33: // post
      {
        mOnlyPostProcessingFlag = true;
        break;
      }

      case 34: // block_size
      {
        try {
          if (std::stoll(optarg) < 1) {
            throw std::invalid_argument("--block_size");
          }
          mBlockSize = std::stoull(optarg);
        } catch (...) {
          printUsage();
          Logger::errorAndTerminate(Logger::wordWrapString(kErrFmtInvalidBlockSize, " ", errorLineIndent));
        }
        break;
      }

      case 35: // no_overlap
      {
        mNoCompressionOverlapFlag = true;
        break;
      }

      case 36: // 40-bit_complex
      {
        m40bitCompressionFlag = true;
        break;
      }

      // unknown parameter or a missing mandatory argument
      case ':':
      case '?': {
        switch (optopt) {
          case 'i': {
            printUsage();
            Logger::errorAndTerminate(Logger::wordWrapString(kErrFmtNoInputFile, " ", errorLineIndent));
            break;
          }
          case 'o': {
            printUsage();
            Logger::errorAndTerminate(Logger::wordWrapString(kErrFmtNoOutputFile, " ", errorLineIndent));
            break;
          }

          case 'r': {
            printUsage();
            Logger::errorAndTerminate(Logger::wordWrapString(kErrFmtNoProgressPrintInterval, " ", errorLineIndent));
            break;
          }

          case 'c': {
            printUsage();
            Logger::errorAndTerminate(Logger::wordWrapString(kErrFmtNoCompressionLevel, " ", errorLineIndent));
            break;
          }

#ifdef _OPENMP
          case 't': {
            printUsage();
            Logger::errorAndTerminate(Logger::wordWrapString(kErrFmtInvalidNumberOfThreads, " ", errorLineIndent));
            break;
          }
#endif

          case 'g': {
            printUsage();
            Logger::errorAndTerminate(Logger::wordWrapString(kErrFmtNoDeviceIndex, " ", errorLineIndent));
            break;
          }

          case 's': {
            printUsage();
            Logger::errorAndTerminate(Logger::wordWrapString(kErrFmtNoSamplingStartTimeStep, " ", errorLineIndent));
            break;
          }

          case 1: // benchmark
          {
            printUsage();
            Logger::errorAndTerminate(Logger::wordWrapString(kErrFmtNoBenchmarkTimeStep, " ", errorLineIndent));
            break;
          }

          case 3: // checkpoint_file
          {
            printUsage();
            Logger::errorAndTerminate(Logger::wordWrapString(kErrFmtNoCheckpointFile, " ", errorLineIndent));
            break;
          }

          case 4: // checkpoint_interval
          {
            printUsage();
            Logger::errorAndTerminate(Logger::wordWrapString(kErrFmtNoCheckpointInterval, " ", errorLineIndent));
            break;
          }

          case 5: // checkpoint_timesteps
          {
            printUsage();
            Logger::errorAndTerminate(Logger::wordWrapString(kErrFmtNoCheckpointTimeSteps, " ", errorLineIndent));
            break;
          }

          case 6: // verbose
          {
            printUsage();
            Logger::errorAndTerminate(Logger::wordWrapString(kErrFmtNoVerboseLevel, " ", errorLineIndent));
            break;
          }

          default: {
            printUsage();
            Logger::errorAndTerminate(Logger::wordWrapString(kErrFmtUnknownParameterOrArgument, " ", errorLineIndent));
            break;
          }
        }
      }

      default: {
        printUsage();
        Logger::errorAndTerminate(Logger::wordWrapString(kErrFmtUnknownParameter, " ", errorLineIndent));
        break;
      }
    }
  }

  if (mPrintVersionFlag)
    return;

  //-- Post checks --//
  if (mInputFileName == "") {
    printUsage();
    Logger::errorAndTerminate(Logger::wordWrapString(kErrFmtNoInputFile, " ", errorLineIndent));
  }

  if (mOutputFileName == "") {
    printUsage();
    Logger::errorAndTerminate(Logger::wordWrapString(kErrFmtNoOutputFile, " ", errorLineIndent));
  }

  if (checkpointFlag) {
    if (mCheckpointFileName == "") {
      printUsage();
      Logger::errorAndTerminate(Logger::wordWrapString(kErrFmtNoCheckpointFile, " ", errorLineIndent));
    }

    if ((mCheckpointInterval <= 0) && (mCheckpointTimeSteps <= 0)) {
      printUsage();
      Logger::errorAndTerminate(Logger::wordWrapString(kErrFmtNoCheckpointIntervalOrTimeSteps, " ", errorLineIndent));
    }
  }

  // Check mOnlyPostProcessingFlag not compatible flags
  if (mOnlyPostProcessingFlag
      && (checkpointFlag || mStorePressureRawFlag || mStorePressureCFlag
          || mStorePressureRmsFlag || mStorePressureMaxFlag || mStorePressureMinFlag
          || mStorePressureMaxAllFlag || mStorePressureMinAllFlag || mStorePressureFinalAllFlag
          || mStoreVelocityRawFlag || mStoreVelocityCFlag
          || mStoreVelocityNonStaggeredCFlag || mStoreVelocityNonStaggeredRawFlag
          || mStoreVelocityRmsFlag || mStoreVelocityMaxFlag || mStoreVelocityMinFlag
          || mStoreVelocityMaxAllFlag || mStoreVelocityMinAllFlag || mStoreVelocityFinalAllFlag)) {
    printUsage();
    Logger::errorAndTerminate(Logger::wordWrapString(kErrFmtInvalidPostProcessingFlag, " ", errorLineIndent));
  }

  if (mOnlyPostProcessingFlag
      && (!mStoreIntensityAvgFlag && !mStoreIntensityAvgCFlag && !mStoreQTermFlag && !mStoreQTermCFlag)) {
    printUsage();
    Logger::errorAndTerminate(Logger::wordWrapString(kErrFmtInvalidPostProcessingFlag, " ", errorLineIndent));
  }

  // set a default flag if necessary
  if (!(mStorePressureRawFlag || mStorePressureCFlag || mStorePressureRmsFlag || mStorePressureMaxFlag || mStorePressureMinFlag || mStorePressureMaxAllFlag || mStorePressureMinAllFlag || mStorePressureFinalAllFlag || mStoreVelocityRawFlag || mStoreVelocityCFlag || mStoreVelocityNonStaggeredCFlag || mStoreVelocityNonStaggeredRawFlag || mStoreVelocityRmsFlag || mStoreVelocityMaxFlag || mStoreVelocityMinFlag || mStoreVelocityMaxAllFlag || mStoreVelocityMinAllFlag || mStoreVelocityFinalAllFlag || mStoreIntensityAvgFlag || mStoreIntensityAvgCFlag || mStoreQTermFlag || mStoreQTermCFlag || mOnlyPostProcessingFlag)) {
    mStorePressureRawFlag = false; //true;
  }
} // end of parseCommandLine
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------ Protected methods -------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Constructor.
 */
CommandLineParameters::CommandLineParameters() : mInputFileName(""), mOutputFileName(""), mCheckpointFileName(""),
#ifdef _OPENMP
                                                 mNumberOfThreads(omp_get_num_procs()),
#else
                                                 mNumberOfThreads(1),
#endif

                                                 mCudaDeviceIdx(CudaParameters::kDefaultDeviceIdx),
                                                 mProgressPrintInterval(kDefaultProgressPrintInterval),
                                                 mCompressionLevel(kDefaultCompressionLevel),
                                                 mBenchmarkFlag(false), mBenchmarkTimeStepCount(0),
                                                 mCheckpointInterval(0), mCheckpointTimeSteps(0),
                                                 mPrintVersionFlag(false),

                                                 // output flags
                                                 mStorePressureRawFlag(false), mStorePressureCFlag(false), mStorePressureRmsFlag(false),
                                                 mStorePressureMaxFlag(false), mStorePressureMinFlag(false),
                                                 mStorePressureMaxAllFlag(false), mStorePressureMinAllFlag(false), mStorePressureFinalAllFlag(false),
                                                 mStoreVelocityRawFlag(false), mStoreVelocityCFlag(false),
                                                 mStoreVelocityNonStaggeredRawFlag(false), mStoreVelocityNonStaggeredCFlag(false),
                                                 mStoreVelocityRmsFlag(false), mStoreVelocityMaxFlag(false), mStoreVelocityMinFlag(false),
                                                 mStoreVelocityMaxAllFlag(false), mStoreVelocityMinAllFlag(false), mStoreVelocityFinalAllFlag(false),
                                                 mStoreIntensityAvgFlag(false), mStoreIntensityAvgCFlag(false), mStoreQTermFlag(false), mStoreQTermCFlag(false), mOnlyPostProcessingFlag(false),
                                                 mCopySensorMaskFlag(false), mSamplingStartTimeStep(0) {

} // end of constructor
//----------------------------------------------------------------------------------------------------------------------
