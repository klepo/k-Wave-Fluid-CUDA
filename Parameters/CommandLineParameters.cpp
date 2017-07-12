/**
 * @file        CommandLineParameters.cpp
 *
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The implementation file containing the command line parameters.
 *
 * @version     kspaceFirstOrder3D 3.4
 *
 * @date        29 August   2012, 11:25 (created) \n
 *              12 July     2017, 15:34 (revised)
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

//Linux build
#ifdef __linux__
  #include <getopt.h>
#endif

//Windows build
#ifdef _WIN64
  #include <GetoptWin64/Getopt.h>
#endif

#ifdef _OPENMP
  #include <omp.h>
#endif


#include <stdexcept>

#include <Logger/Logger.h>
#include <Parameters/CudaParameters.h>
#include <Parameters/CommandLineParameters.h>
#include <HDF5/HDF5_File.h>

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
void CommandLineParameters::printUsage()
{
  TLogger::Log(TLogger::TLogLevel::BASIC, OUT_FMT_USAGE_PART_1);

  #ifdef _OPENMP
    TLogger::Log(TLogger::TLogLevel::BASIC,
                 OUT_FMT_USAGE_THREADS,
                 omp_get_num_procs());
  #endif

  TLogger::Log(TLogger::TLogLevel::BASIC,
               OUT_FMT_USAGE_PART_2,
               kDefaultProgressPrintInterval,
               kDefaultCompressionLevel);
}// end of printUsage
//----------------------------------------------------------------------------------------------------------------------

/**
 * Print out commandline parameters.
 */
void CommandLineParameters::printComandlineParamers()
{
  TLogger::Log(TLogger::TLogLevel::ADVANCED, OUT_FMT_SEPARATOR);

  TLogger::Log(TLogger::TLogLevel::ADVANCED,
               TLogger::WordWrapString(OUT_FMT_INPUT_FILE + mInputFileName,
                                       ERR_FMT_PATH_DELIMITERS,
                                       15).c_str());

  TLogger::Log(TLogger::TLogLevel::ADVANCED,
               TLogger::WordWrapString(OUT_FMT_OUTPUT_FILE + mOutputFileName,
                                       ERR_FMT_PATH_DELIMITERS,
                                       15).c_str());

  if (isCheckpointEnabled())
  {
    TLogger::Log(TLogger::TLogLevel::ADVANCED,
                 TLogger::WordWrapString(OUT_FMT_CHECKPOINT_FILE + mCheckpointFileName,
                                         ERR_FMT_PATH_DELIMITERS,
                                         15).c_str());

    TLogger::Log(TLogger::TLogLevel::ADVANCED, OUT_FMT_SEPARATOR);

    TLogger::Log(TLogger::TLogLevel::ADVANCED, OUT_FMT_CHECKPOINT_INTERVAL, mCheckpointInterval);
  }
  else
  {
    TLogger::Log(TLogger::TLogLevel::ADVANCED, OUT_FMT_SEPARATOR);
  }


  TLogger::Log(TLogger::TLogLevel::ADVANCED, OUT_FMT_COMPRESSION_LEVEL, mCompressionLevel);

  TLogger::Log(TLogger::TLogLevel::FULL,     OUT_FMT_PRINT_PROGRESS_INTERVAL, mProgressPrintInterval);

  if (mBenchmarkFlag)
  {
    TLogger::Log(TLogger::TLogLevel::FULL, OUT_FMT_BENCHMARK_TIME_STEP, mBenchmarkTimeStepCount);
  }

  TLogger::Log(TLogger::TLogLevel::ADVANCED, OUT_FMT_SAMPLING_FLAGS);


  string sampledQuantitiesList = "";
  // Sampled p quantities

  if (mStorePressureRaw)
  {
    sampledQuantitiesList += "p_raw, ";
  }
  if (mStorePressureRms)
  {
    sampledQuantitiesList += "p_rms, ";
  }
  if (mStorePressureMax)
  {
    sampledQuantitiesList += "p_max, ";
  }
  if (mStorePressureMin)
  {
    sampledQuantitiesList += "p_min, ";
  }
  if (mStorePressureMaxAllDomain)
  {
    sampledQuantitiesList += "p_max_all, ";
  }
  if (mStorePressureMinAllDomain)
  {
    sampledQuantitiesList += "p_min_all, ";
  }
  if (mStorePressureFinalAllDomain)
  {
    sampledQuantitiesList += "p_final, ";
  }

  // Sampled u quantities
  if (mStoreVelocityRaw)
  {
    sampledQuantitiesList += "u_raw, ";
  }
  if (mStoreVelocityRms)
  {
    sampledQuantitiesList += "u_rms, ";
  }
  if (mStoreVelocityMax)
  {
    sampledQuantitiesList += "u_max, ";
  }
  if (mStoreVelocityMin)
  {
    sampledQuantitiesList += "u_min, ";
  }
  if (mStoreVelocityMaxAllDomain)
  {
    sampledQuantitiesList += "u_max_all, ";
  }
  if (mStoreVelocityMinAllDomain)
  {
    sampledQuantitiesList += "u_min_all, ";
  }
  if (mStoreVelocityFinalAllDomain)
  {
    sampledQuantitiesList += "u_final, ";
  }

  if (mStoreVelocityNonStaggeredRaw)
  {
    sampledQuantitiesList += "u_non_staggered_raw, ";
  }

  // remove comma and space symbols
  if (sampledQuantitiesList.length() > 0)
  {
    sampledQuantitiesList.pop_back();
    sampledQuantitiesList.pop_back();
  }

  TLogger::Log(TLogger::TLogLevel::ADVANCED,
               TLogger::WordWrapString(sampledQuantitiesList,
                                       " ",2).c_str());

  TLogger::Log(TLogger::TLogLevel::ADVANCED, OUT_FMT_SEPARATOR);

  TLogger::Log(TLogger::TLogLevel::ADVANCED, OUT_FMT_SAMPLING_BEGINS_AT, mSamplingStartTimeStep + 1);

  if (mCopySensorMask)
  {
    TLogger::Log(TLogger::TLogLevel::ADVANCED, OUT_FMT_COPY_SENSOR_MASK);
  }
}// end of printComandlineParamers
//----------------------------------------------------------------------------------------------------------------------

/**
 * Parse command line.
 */
void CommandLineParameters::parseCommandLine(int argc, char** argv)
{
  char c;
  int  longIndex = -1;
  bool checkpointFlag = false;

  constexpr int errorLineIndentation = 9;

  // all optional arguments are in fact requested. This was chosen to prevent
  // getopt error messages and provide custom error handling.
  #ifdef _OPENMP
    const char* shortOpts = "i:o:r:c:t:g:puhs:";
  #else
    const char* shortOpts = "i:o:r:c:g:puhs:";
  #endif

  const struct option longOpts[] =
  {
    { "benchmark",            required_argument, nullptr, 1 },
    { "copy_sensor_mask",     no_argument,       nullptr, 2 },
    { "checkpoint_file"    ,  required_argument, nullptr, 3 },
    { "checkpoint_interval",  required_argument, nullptr, 4 },
    { "help",                 no_argument, nullptr,      'h'},
    { "verbose",              required_argument, nullptr, 5 },
    { "version",              no_argument, nullptr,       6 },

    { "p_raw",                no_argument, nullptr,'p' },
    { "p_rms",                no_argument, nullptr, 10 },
    { "p_max",                no_argument, nullptr, 11 },
    { "p_min",                no_argument, nullptr, 12 },
    { "p_max_all",            no_argument, nullptr, 13 },
    { "p_min_all",            no_argument, nullptr, 14 },
    { "p_final",              no_argument, nullptr, 15 },

    { "u_raw",                no_argument, nullptr,'u' },
    { "u_rms",                no_argument, nullptr, 20},
    { "u_max",                no_argument, nullptr, 21},
    { "u_min",                no_argument, nullptr, 22},
    { "u_max_all",            no_argument, nullptr, 23},
    { "u_min_all",            no_argument, nullptr, 24},
    { "u_final",              no_argument, nullptr, 25},
    { "u_non_staggered_raw",  no_argument, nullptr, 26},

    { nullptr,                no_argument, nullptr, 0}
  };

  // all optional arguments are in fact requested. This was chosen to prevent
  // getopt error messages and provide custom error handling.
  opterr = 0;

  // Short parameters //
  while ((c = getopt_long (argc, argv, shortOpts, longOpts, &longIndex )) != -1)
  {
    switch (c)
    {
      case 'i':
      {
        // test if the wile was correctly entered (if not, getopt could eat
        // the following parameter)
        if ((optarg != nullptr) &&
            ((strlen(optarg) > 0) && (optarg[0] != '-')))
        {
          mInputFileName = optarg;
        }
        else
        {
          printUsage();
          TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_INPUT_FILE,
                                                             " ",
                                                             errorLineIndentation).c_str());
        }
        break;
      }

      case 'o':
      {
        // test if the wile was correctly entered (if not, getopt could eat
        // the following parameter)
        if ((optarg != nullptr) &&
            ((strlen(optarg) > 0) && (optarg[0] != '-')))
        {
          mOutputFileName = optarg;
        }
        else
        {
          printUsage();
          TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_OUTPUT_FILE,
                                                             " ",
                                                             errorLineIndentation).c_str());
        }
        break;
      }

      case 'r':
      {
        try
        {
          int convertedValue = std::stoi(optarg);
          if ((convertedValue  < 1) || (convertedValue  > 100))
          {
            throw std::invalid_argument("-r");
          }
          mProgressPrintInterval = std::stoll(optarg);
        }
        catch (...)
        {
          printUsage();
          TLogger::ErrorAndTerminate(TLogger::WordWrapString(FMT_NO_PROGRESS_PRINT_INTERVAL,
                                                             " ",
                                                             errorLineIndentation).c_str());
        }
        break;
      }

  #ifdef _OPENMP
      case 't':
      {
        try
        {
          if (std::stoi(optarg) < 1)
          {
            throw std::invalid_argument("-t");
          }
          mNumberOfThreads = std::stoll(optarg);
        }
        catch (...)
        {
          printUsage();
          TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_THREAD_NUMBER,
                                                             " ",
                                                             errorLineIndentation).c_str());
        }
        break;
      }
  #endif
      case 'g':
      {
        try
        {
          mCudaDeviceIdx = std::stoi(optarg);
          if (mCudaDeviceIdx < 0)
          {
            throw std::invalid_argument("-g");
          }
        }
        catch (...)
        {
          printUsage();
          TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_GPU_NUMBER,
                                                             " ",
                                                             errorLineIndentation).c_str());
        }
        break;
      }

      case 'c':
      {
        try
        {
          int covertedValue = std::stoi(optarg);
          if ((covertedValue < 0) || (covertedValue > 9))
          {
            throw std::invalid_argument("-c");
          }
          mCompressionLevel = std::stoll(optarg);
        }
        catch (...)
        {
          printUsage();
          TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_COMPRESSION_LEVEL,
                                                             " ",
                                                             errorLineIndentation).c_str());
        }
        break;
      }

      case 'h':
      {
        printUsage();
        exit(EXIT_SUCCESS);
      }

      case 's':
      {
        try
        {
          if (std::stoll(optarg) < 1)
          {
            throw std::invalid_argument("-s");
          }
          mSamplingStartTimeStep = std::stoll(optarg) - 1;
        }
        catch (...)
        {
          printUsage();
          TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_START_TIME_STEP,
                                                             " ",
                                                             errorLineIndentation).c_str());
        }
        break;
      }

      case 1: // benchmark
      {
        try
        {
          mBenchmarkFlag = true;
          if (std::stoll(optarg) <= 0)
          {
            throw std::invalid_argument("benchmark");
          }
          mBenchmarkTimeStepCount = std::stoll(optarg);
        }
        catch (...)
        {
          printUsage();
          TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_BENCHMARK_STEP_SET,
                                                             " ",
                                                             errorLineIndentation).c_str());
        }
        break;
      }

      case 2: // copy_sensor_mask
      {
        mCopySensorMask = true;
        break;
      }

      case 3: // checkpoint_file
      {
        checkpointFlag = true;
        // test if the wile was correctly entered (if not, getopt could eat
        // the following parameter)
        if ((optarg != NULL) &&
            ((strlen(optarg) > 0) && (optarg[0] != '-')))
        {
          mCheckpointFileName = optarg;
        }
        else
        {
          printUsage();
          TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_CHECKPOINT_FILE,
                                                             " ",
                                                             errorLineIndentation).c_str());
        }
        break;
      }

      case 4: // checkpoint_interval
      {
        try
        {
          checkpointFlag = true;
          if (std::stoll(optarg) <= 0)
          {
           throw std::invalid_argument("checkpoint_interval");
          }
          mCheckpointInterval = std::stoll(optarg);
        }
        catch (...)
        {
          printUsage();
          TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_CHECKPOINT_INTERVAL,
                                                             " ",
                                                             errorLineIndentation).c_str());
        }
        break;
      }

      case 5: // verbose
      {
        try
        {
          int verboseLevel = std::stoi(optarg);
          if ((verboseLevel < 0) || (verboseLevel > 2))
          {
            throw std::invalid_argument("verbose");
          }
          TLogger::SetLevel(static_cast<TLogger::TLogLevel> (verboseLevel));
        }
        catch (...)
        {
          printUsage();
          TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_VERBOSE_LEVEL,
                                                             " ",
                                                             errorLineIndentation).c_str());
        }
        break;
      }

      case 6: // version
      {
        mPrintVersionFlag = true;
        break;
      }

      case 'p':
      {
        mStorePressureRaw = true;
        break;
      }

      case 10: // p_rms
      {
        mStorePressureRms = true;
        break;
      }

      case 11: // p_max
      {
        mStorePressureMax = true;
        break;
      }

      case 12: // p_min
      {
        mStorePressureMin = true;
        break;
      }

      case 13: // p_max_all
      {
        mStorePressureMaxAllDomain = true;
        break;
      }

      case 14: // p_min_all
      {
        mStorePressureMinAllDomain = true;
        break;
      }

      case 15: // p_final
      {
        mStorePressureFinalAllDomain = true;
        break;
      }

      case 'u':
      {
        mStoreVelocityRaw = true;
        break;
      }

      case 20: // u_rms
      {
        mStoreVelocityRms = true;
        break;
      }

      case 21: // u_max
      {
        mStoreVelocityMax = true;
        break;
      }

      case 22: // u_min
      {
        mStoreVelocityMin = true;
        break;
      }

      case 23: // u_max_all
      {
        mStoreVelocityMaxAllDomain = true;
        break;
      }

      case 24: // u_min_all
      {
        mStoreVelocityMinAllDomain = true;
        break;
      }

      case 25: // u_final
      {
        mStoreVelocityFinalAllDomain = true;
        break;
      }

      case 26: // u_non_staggered_raw
      {
        mStoreVelocityNonStaggeredRaw = true;
        break;
      }

      // unknown parameter or a missing mandatory argument
      case ':':
      case '?':
      {
        switch (optopt)
        {
          case 'i':
          {
            printUsage();
            TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_INPUT_FILE,
                                                               " ",
                                                               errorLineIndentation).c_str());
            break;
          }
          case 'o':
          {
            printUsage();
            TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_OUTPUT_FILE,
                                                               " ",
                                                               errorLineIndentation).c_str());
            break;
          }

          case 'r':
          {
            printUsage();
            TLogger::ErrorAndTerminate(TLogger::WordWrapString(FMT_NO_PROGRESS_PRINT_INTERVAL,
                                                               " ",
                                                               errorLineIndentation).c_str());
            break;
          }

          case 'c':
          {
            printUsage();
            TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_COMPRESSION_LEVEL,
                                                               " ",
                                                               errorLineIndentation).c_str());
            break;
          }

        #ifdef _OPENMP
          case 't':
          {
            printUsage();
            TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_THREAD_NUMBER,
                                                               " ",
                                                               errorLineIndentation).c_str());
            break;
          }
        #endif

          case 'g':
          {
            printUsage();
            TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_GPU_NUMBER,
                                                               " ",
                                                               errorLineIndentation).c_str());
            break;
          }

          case 's':
          {
            printUsage();
            TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_START_TIME_STEP,
                                                               " ",
                                                               errorLineIndentation).c_str());
            break;
          }

          case 1: // benchmark
          {
            printUsage();
            TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_BENCHMARK_STEP_SET,
                                                               " ",
                                                               errorLineIndentation).c_str());
            break;
          }

          case 3: // checkpoint_file
          {
            printUsage();
            TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_CHECKPOINT_FILE,
                                                               " ",
                                                               errorLineIndentation).c_str());
            break;
          }

          case 4: // checkpoint_interval
          {
            printUsage();
            TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_CHECKPOINT_INTERVAL,
                                                               " ",
                                                               errorLineIndentation).c_str());
            break;
          }

          case 5: // verbose
          {
            printUsage();
            TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_VERBOSE_LEVEL,
                                                               " ",
                                                               errorLineIndentation).c_str());
            break;
          }

          default :
          {
            printUsage();
            TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_UNKNOW_PARAMETER_OR_ARGUMENT,
                                                               " ",
                                                               errorLineIndentation).c_str());
            break;
          }
        }
      }

      default:
      {
        printUsage();
        TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_UNKNOWN_PARAMETER,
                                                           " ",
                                                           errorLineIndentation).c_str());
        break;
      }
    }
  }

  if (mPrintVersionFlag) return;

  //-- Post checks --//
  if (mInputFileName == "")
  {
    printUsage();
    TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_INPUT_FILE,
                                                       " ",
                                                       errorLineIndentation).c_str());
  }

  if (mOutputFileName == "")
  {
    printUsage();
    TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_OUTPUT_FILE,
                                                       " ",
                                                       errorLineIndentation).c_str());
  }

  if (checkpointFlag)
  {
    if (mCheckpointFileName == "")
    {
      printUsage();
      TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_CHECKPOINT_FILE,
                                                         " ",
                                                         errorLineIndentation).c_str());
    }

    if (mCheckpointInterval <= 0)
    {
      printUsage();
      TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_CHECKPOINT_INTERVAL,
                                                         " ",
                                                         errorLineIndentation).c_str());
    }
  }

  // set a default flag if necessary
  if (!(mStorePressureRaw     || mStorePressureRms     || mStorePressureMax   || mStorePressureMin ||
        mStorePressureMaxAllDomain || mStorePressureMinAllDomain || mStorePressureFinalAllDomain ||
        mStoreVelocityRaw     || mStoreVelocityNonStaggeredRaw        ||
        mStoreVelocityRms     || mStoreVelocityMax     || mStoreVelocityMin   ||
        mStoreVelocityMaxAllDomain || mStoreVelocityMinAllDomain || mStoreVelocityFinalAllDomain ))
  {
    mStorePressureRaw = true;
  }
}// end of parseCommandLine
//----------------------------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------ Protected methods -------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Constructor.
 */
CommandLineParameters::CommandLineParameters() :
  mInputFileName(""), mOutputFileName (""), mCheckpointFileName(""),
  #ifdef _OPENMP
    mNumberOfThreads(omp_get_num_procs()),
  #else
    mNumberOfThreads(1),
  #endif

  mCudaDeviceIdx(CudaParameters::kDefaultDeviceIdx),
  mProgressPrintInterval(kDefaultProgressPrintInterval),
  mCompressionLevel(kDefaultCompressionLevel),
  mBenchmarkFlag (false), mBenchmarkTimeStepCount(0),
  mCheckpointInterval(0),
  mPrintVersionFlag (false),

  // output flags
  mStorePressureRaw(false), mStorePressureRms(false), mStorePressureMax(false), mStorePressureMin(false),
  mStorePressureMaxAllDomain(false), mStorePressureMinAllDomain(false), mStorePressureFinalAllDomain(false),
  mStoreVelocityRaw(false), mStoreVelocityNonStaggeredRaw(false),
  mStoreVelocityRms(false), mStoreVelocityMax(false), mStoreVelocityMin(false),
  mStoreVelocityMaxAllDomain(false), mStoreVelocityMinAllDomain(false), mStoreVelocityFinalAllDomain(false),
  mCopySensorMask(false),
  mSamplingStartTimeStep(0)
{

}// end of CommandLineParameters
//----------------------------------------------------------------------------------------------------------------------
