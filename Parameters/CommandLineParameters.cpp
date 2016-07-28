/**
 * @file        CommandLineParameters.cpp
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
 *              18 July     2016, 13:56 (revised)
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

#include <cstring>
#include <limits>
#include <stdexcept>

#ifdef _OPENMP
  #include <omp.h>
#endif

#include <Logger/ErrorMessages.h>
#include <Logger/Logger.h>
#include <Parameters/CommandLineParameters.h>
#include <HDF5/HDF5_File.h>

using std::string;

//------------------------------------------------------------------------------------------------//
//------------------------------------------ CONSTANTS -------------------------------------------//
//------------------------------------------------------------------------------------------------//



//------------------------------------------------------------------------------------------------//
//--------------------------------------- Public methods -----------------------------------------//
//------------------------------------------------------------------------------------------------//

/**
 * Constructor.
 */
TCommandLineParameters::TCommandLineParameters() :
        inputFileName(""), outputFileName (""), checkpointFileName(""),
        #ifdef _OPENMP
          numberOfThreads(omp_get_num_procs()),
        #else
          numberOfThreads(1),
        #endif

        cudaDeviceIdx(-1), // default is undefined -1

        progressPrintInterval(DEFAULT_PROGRESS_PRINT_INTERVAL),
        compressionLevel(DEFAULT_COMPRESSION_LEVEL),
        benchmarkFlag (false), benchmarkTimeStepCount(0),
        checkpointInterval(0),
        printVersion (false),

        store_p_raw(false), store_p_rms(false), store_p_max(false), store_p_min(false),
        store_p_max_all(false), store_p_min_all(false), store_p_final(false),
        store_u_raw(false), store_u_non_staggered_raw(false),
        store_u_rms(false), store_u_max(false), store_u_min(false),
        store_u_max_all(false), store_u_min_all(false), store_u_final(false),
        copySensorMask(false),
        startTimeStep(0)
{

}// end of TCommandLineParameters
//--------------------------------------------------------------------------------------------------

/**
 * Print usage.
 */
void TCommandLineParameters::PrintUsage()
{
  TLogger::Log(TLogger::BASIC, OUT_FMT_USAGE_PART_1);

  #ifdef _OPENMP
    TLogger::Log(TLogger::BASIC,
                 OUT_FMT_USAGE_THREADS,
                 omp_get_num_procs());
  #endif

  TLogger::Log(TLogger::BASIC,
               OUT_FMT_USAGE_PART_2,
               DEFAULT_PROGRESS_PRINT_INTERVAL,
               DEFAULT_COMPRESSION_LEVEL);
}// end of PrintUsage
//--------------------------------------------------------------------------------------------------

/**
 * Print out commandline parameters.
 */
void TCommandLineParameters::PrintComandlineParamers()
{
  TLogger::Log(TLogger::ADVANCED, OUT_FMT_SEPARATOR);

  TLogger::Log(TLogger::ADVANCED,
               TLogger::WordWrapString(OUT_FMT_INPUT_FILE + inputFileName,
                                       ERR_FMT_PATH_DELIMITERS,
                                       15).c_str());

  TLogger::Log(TLogger::ADVANCED,
               TLogger::WordWrapString(OUT_FMT_OUTPUT_FILE + outputFileName,
                                       ERR_FMT_PATH_DELIMITERS,
                                       15).c_str());

  if (IsCheckpointEnabled())
  {
    TLogger::Log(TLogger::ADVANCED,
                 TLogger::WordWrapString(OUT_FMT_CHECKPOINT_FILE + checkpointFileName,
                                         ERR_FMT_PATH_DELIMITERS,
                                         15).c_str());

    TLogger::Log(TLogger::ADVANCED, OUT_FMT_SEPARATOR);

    TLogger::Log(TLogger::ADVANCED, OUT_FMT_CHECKPOINT_INTERVAL, checkpointInterval);
  }
  else
  {
    TLogger::Log(TLogger::ADVANCED, OUT_FMT_SEPARATOR);
  }


  TLogger::Log(TLogger::ADVANCED, OUT_FMT_COMPRESSION_LEVEL, compressionLevel);

  TLogger::Log(TLogger::FULL,     OUT_FMT_PRINT_PROGRESS_INTERVAL, progressPrintInterval);

  if (benchmarkFlag)
  {
    TLogger::Log(TLogger::FULL, OUT_FMT_BENCHMARK_TIME_STEP, benchmarkTimeStepCount);
  }

  TLogger::Log(TLogger::ADVANCED, OUT_FMT_SAMPLING_FLAGS);


  string sampledQuantitiesList = "";
  // Sampled p quantities

  if (store_p_raw)
  {
    sampledQuantitiesList += "p_raw, ";
  }
  if (store_p_rms)
  {
    sampledQuantitiesList += "p_rms, ";
  }
  if (store_p_max)
  {
    sampledQuantitiesList += "p_max, ";
  }
  if (store_p_min)
  {
    sampledQuantitiesList += "p_min, ";
  }
  if (store_p_max_all)
  {
    sampledQuantitiesList += "p_max_all, ";
  }
  if (store_p_min_all)
  {
    sampledQuantitiesList += "p_min_all, ";
  }
  if (store_p_final)
  {
    sampledQuantitiesList += "p_final, ";
  }

  // Sampled u quantities
  if (store_u_raw)
  {
    sampledQuantitiesList += "u_raw, ";
  }
  if (store_u_rms)
  {
    sampledQuantitiesList += "u_rms, ";
  }
  if (store_u_max)
  {
    sampledQuantitiesList += "u_max, ";
  }
  if (store_u_min)
  {
    sampledQuantitiesList += "u_min, ";
  }
  if (store_u_max_all)
  {
    sampledQuantitiesList += "u_max_all, ";
  }
  if (store_u_min_all)
  {
    sampledQuantitiesList += "u_min_all, ";
  }
  if (store_u_final)
  {
    sampledQuantitiesList += "u_final, ";
  }

  if (store_u_non_staggered_raw)
  {
    sampledQuantitiesList += "u_non_staggered_raw, ";
  }

  // remove comma and space symbols
  if (sampledQuantitiesList.length() > 0)
  {
    sampledQuantitiesList.pop_back();
    sampledQuantitiesList.pop_back();
  }

  TLogger::Log(TLogger::ADVANCED,
               TLogger::WordWrapString(sampledQuantitiesList,
                                       " ",2).c_str());

  TLogger::Log(TLogger::ADVANCED, OUT_FMT_SEPARATOR);

  TLogger::Log(TLogger::ADVANCED, OUT_FMT_SAMPLING_BEGINS_AT, startTimeStep + 1);

  if (copySensorMask)
  {
    TLogger::Log(TLogger::ADVANCED, OUT_FMT_COPY_SENSOR_MASK);
  }
}// end of PrintComandlineParamers
//--------------------------------------------------------------------------------------------------

/**
 * Parse command line.
 * @param [in, out] argc
 * @param [in, out] argv
 */
void TCommandLineParameters::ParseCommandLine(int argc, char** argv)
{
  char c;
  int longIndex = -1;
  bool checkpointFlag = false;

  const int errorLineIndentation = 9;

  // all optional arguments are in fact requested. This was chosen to prevent
  // getopt error messages and provide custom error handling.
  #ifdef _OPENMP
    const char * shortOpts = "i:o:r:c:t:g:puhs:";
  #else
    const char * shortOpts = "i:o:r:c:g:puhs:";
  #endif

  const struct option longOpts[] =
  {
    { "benchmark",            required_argument, NULL, 1 },
    { "copy_sensor_mask",     no_argument,       NULL, 2 },
    { "checkpoint_file"    ,  required_argument, NULL, 3 },
    { "checkpoint_interval",  required_argument, NULL, 4 },
    { "help",                 no_argument, NULL,      'h'},
    { "verbose",              required_argument, NULL, 5 },
    { "version",              no_argument, NULL,       6 },

    { "p_raw",                no_argument, NULL,'p' },
    { "p_rms",                no_argument, NULL, 10 },
    { "p_max",                no_argument, NULL, 11 },
    { "p_min",                no_argument, NULL, 12 },
    { "p_max_all",            no_argument, NULL, 13 },
    { "p_min_all",            no_argument, NULL, 14 },
    { "p_final",              no_argument, NULL, 15 },

    { "u_raw",                no_argument, NULL,'u' },
    { "u_rms",                no_argument, NULL, 20},
    { "u_max",                no_argument, NULL, 21},
    { "u_min",                no_argument, NULL, 22},
    { "u_max_all",            no_argument, NULL, 23},
    { "u_min_all",            no_argument, NULL, 24},
    { "u_final",              no_argument, NULL, 25},
    { "u_non_staggered_raw",  no_argument, NULL, 26},

    { NULL,                   no_argument, NULL, 0}
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
        if ((optarg != NULL) &&
            ((strlen(optarg) > 0) && (optarg[0] != '-')))
        {
          inputFileName = optarg;
        }
        else
        {
          PrintUsage();
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
        if ((optarg != NULL) &&
            ((strlen(optarg) > 0) && (optarg[0] != '-')))
        {
          outputFileName = optarg;
        }
        else
        {
          PrintUsage();
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
          progressPrintInterval = std::stoll(optarg);
        }
        catch (...)
        {
          PrintUsage();
          TLogger::ErrorAndTerminate(TLogger::WordWrapString(FMT_NO_PROGRESS_PRINT_INTERVAL,
                                                             " ",
                                                             errorLineIndentation).c_str());
        }
        break;
      }

      case 't':
      {
        try
        {
          if (std::stoi(optarg) < 1)
          {
            throw std::invalid_argument("-t");
          }
          numberOfThreads = std::stoll(optarg);
        }
        catch (...)
        {
          PrintUsage();
          TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_THREAD_NUMBER,
                                                             " ",
                                                             errorLineIndentation).c_str());
        }
        break;
      }

      case 'g':
      {
        try
        {
          cudaDeviceIdx = std::stoi(optarg);
          if (cudaDeviceIdx < 0)
          {
            throw std::invalid_argument("-g");
          }
        }
        catch (...)
        {
          PrintUsage();
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
          compressionLevel = std::stoll(optarg);
        }
        catch (...)
        {
          PrintUsage();
          TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_COMPRESSION_LEVEL,
                                                             " ",
                                                             errorLineIndentation).c_str());
        }
        break;
      }

      case 'h':
      {
        PrintUsage();
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
          startTimeStep = std::stoll(optarg) - 1;
        }
        catch (...)
        {
          PrintUsage();
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
          benchmarkFlag = true;
          if (std::stoll(optarg) <= 0)
          {
            throw std::invalid_argument("benchmark");
          }
          benchmarkTimeStepCount = std::stoll(optarg);
        }
        catch (...)
        {
          PrintUsage();
          TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_BENCHMARK_STEP_SET,
                                                             " ",
                                                             errorLineIndentation).c_str());
        }
        break;
      }

      case 2: // copy_sensor_mask
      {
        copySensorMask = true;
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
          checkpointFileName = optarg;
        }
        else
        {
          PrintUsage();
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
          checkpointInterval = std::stoll(optarg);
        }
        catch (...)
        {
          PrintUsage();
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
          PrintUsage();
          TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_VERBOSE_LEVEL,
                                                             " ",
                                                             errorLineIndentation).c_str());
        }
        break;
      }

      case 6: // version
      {
        printVersion = true;
        break;
      }

      case 'p':
      {
        store_p_raw = true;
        break;
      }

      case 10: // p_rms
      {
        store_p_rms = true;
        break;
      }

      case 11: // p_max
      {
        store_p_max = true;
        break;
      }

      case 12: // p_min
      {
        store_p_min = true;
        break;
      }

      case 13: // p_max_all
      {
        store_p_max_all = true;
        break;
      }

      case 14: // p_min_all
      {
        store_p_min_all = true;
        break;
      }

      case 15: // p_final
      {
        store_p_final = true;
        break;
      }

      case 'u':
      {
        store_u_raw = true;
        break;
      }

      case 20: // u_rms
      {
        store_u_rms = true;
        break;
      }

      case 21: // u_max
      {
        store_u_max = true;
        break;
      }

      case 22: // u_min
      {
        store_u_min = true;
        break;
      }

      case 23: // u_max_all
      {
        store_u_max_all = true;
        break;
      }

      case 24: // u_min_all
      {
        store_u_min_all = true;
        break;
      }

      case 25: // u_final
      {
        store_u_final = true;
        break;
      }

      case 26: // u_non_staggered_raw
      {
        store_u_non_staggered_raw = true;
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
            PrintUsage();
            TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_INPUT_FILE,
                                                               " ",
                                                               errorLineIndentation).c_str());
            break;
          }
          case 'o':
          {
            PrintUsage();
            TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_OUTPUT_FILE,
                                                               " ",
                                                               errorLineIndentation).c_str());
            break;
          }

          case 'r':
          {
            PrintUsage();
            TLogger::ErrorAndTerminate(TLogger::WordWrapString(FMT_NO_PROGRESS_PRINT_INTERVAL,
                                                               " ",
                                                               errorLineIndentation).c_str());
            break;
          }

          case 'c':
          {
            PrintUsage();
            TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_COMPRESSION_LEVEL,
                                                               " ",
                                                               errorLineIndentation).c_str());
            break;
          }

          case 't':
          {
            PrintUsage();
            TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_THREAD_NUMBER,
                                                               " ",
                                                               errorLineIndentation).c_str());
            break;
          }

          case 'g':
          {
            PrintUsage();
            TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_GPU_NUMBER,
                                                               " ",
                                                               errorLineIndentation).c_str());
            break;
          }

          case 's':
          {
            PrintUsage();
            TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_START_TIME_STEP,
                                                               " ",
                                                               errorLineIndentation).c_str());
            break;
          }

          case 1: // benchmark
          {
            PrintUsage();
            TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_BENCHMARK_STEP_SET,
                                                               " ",
                                                               errorLineIndentation).c_str());
            break;
          }

          case 3: // checkpoint_file
          {
            PrintUsage();
            TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_CHECKPOINT_FILE,
                                                               " ",
                                                               errorLineIndentation).c_str());
            break;
          }

          case 4: // checkpoint_interval
          {
            PrintUsage();
            TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_CHECKPOINT_INTERVAL,
                                                               " ",
                                                               errorLineIndentation).c_str());
            break;
          }

          case 5: // verbose
          {
            PrintUsage();
            TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_VERBOSE_LEVEL,
                                                               " ",
                                                               errorLineIndentation).c_str());
            break;
          }

          default :
          {
            PrintUsage();
            TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_UNKNOW_PARAMETER_OR_ARGUMENT,
                                                               " ",
                                                               errorLineIndentation).c_str());
            break;
          }
        }
      }

      default:
      {
        PrintUsage();
        TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_UNKNOWN_PARAMETER,
                                                           " ",
                                                           errorLineIndentation).c_str());
        break;
      }
    }
  }

  if (printVersion) return;

  //-- Post checks --//
  if (inputFileName == "")
  {
    PrintUsage();
    TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_INPUT_FILE,
                                                       " ",
                                                       errorLineIndentation).c_str());
  }

  if (outputFileName == "")
  {
    PrintUsage();
    TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_OUTPUT_FILE,
                                                       " ",
                                                       errorLineIndentation).c_str());
  }

  if (checkpointFlag)
  {
    if (checkpointFileName == "")
    {
      PrintUsage();
      TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_CHECKPOINT_FILE,
                                                         " ",
                                                         errorLineIndentation).c_str());
    }

    if (checkpointInterval <= 0)
    {
      PrintUsage();
      TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_CHECKPOINT_INTERVAL,
                                                         " ",
                                                         errorLineIndentation).c_str());
    }
  }

  // set a default flag if necessary
  if (!(store_p_raw     || store_p_rms     || store_p_max   || store_p_min ||
        store_p_max_all || store_p_min_all || store_p_final ||
        store_u_raw     || store_u_non_staggered_raw        ||
        store_u_rms     || store_u_max     || store_u_min   ||
        store_u_max_all || store_u_min_all || store_u_final ))
  {
    store_p_raw = true;
  }
}// end of ParseCommandLine
//--------------------------------------------------------------------------------------------------
