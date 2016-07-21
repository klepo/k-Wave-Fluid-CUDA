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
 * @date        29 August   2012, 11:25 (created) \n
 *              18 July     2016, 13:56 (revised)
 *
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

//--------------------------------------------------------------------------//
//-------------------------- Constants -------------------------------------//
//--------------------------------------------------------------------------//

//--------------------------------------------------------------------------//
//--------------------------- Public   -------------------------------------//
//--------------------------------------------------------------------------//

/**
 * Constructor.
 */
TCommandLineParameters::TCommandLineParameters() :
        InputFileName(""), OutputFileName (""), CheckpointFileName(""),
        #ifdef _OPENMP
          NumberOfThreads(omp_get_num_procs()),
        #else
          NumberOfThreads(1),
        #endif

        GPUDeviceIdx(-1), // default is undefined -1

        ProgressPrintInterval(DefaultProgressPrintInterval),
        CompressionLevel(DefaultCompressionLevel),
        BenchmarkFlag (false), BenchmarkTimeStepsCount(0),
        CheckpointInterval(0),
        PrintVersion (false),

        Store_p_raw(false), Store_p_rms(false), Store_p_max(false), Store_p_min(false),
        Store_p_max_all(false), Store_p_min_all(false), Store_p_final(false),
        Store_u_raw(false), Store_u_non_staggered_raw(false),
        Store_u_rms(false), Store_u_max(false), Store_u_min(false),
        Store_u_max_all(false), Store_u_min_all(false), Store_u_final(false),
        CopySensorMask(false),
        StartTimeStep(0)
{

}// end of TCommandLineParameters
//------------------------------------------------------------------------------

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
               DefaultProgressPrintInterval,
               DefaultCompressionLevel);
}// end of PrintUsage
//------------------------------------------------------------------------------

/**
 * Print out commandline parameters.
 */
void TCommandLineParameters::PrintComandlineParamers()
{
  TLogger::Log(TLogger::ADVANCED,OUT_FMT_SEPARATOR);

  TLogger::Log(TLogger::ADVANCED,
               TLogger::WordWrapString(OUT_FMT_INPUT_FILE + InputFileName,
                                       ERR_FMT_PATH_DELIMITERS,
                                       15).c_str());

  TLogger::Log(TLogger::ADVANCED,
               TLogger::WordWrapString(OUT_FMT_OUTPUT_FILE + OutputFileName,
                                       ERR_FMT_PATH_DELIMITERS,
                                       15).c_str());


  if (IsCheckpointEnabled())
  {
    TLogger::Log(TLogger::ADVANCED,
                 TLogger::WordWrapString(OUT_FMT_CHECKPOINT_FILE + CheckpointFileName,
                                         ERR_FMT_PATH_DELIMITERS,
                                         15).c_str());

    TLogger::Log(TLogger::ADVANCED,OUT_FMT_SEPARATOR);

    TLogger::Log(TLogger::ADVANCED,
                 OUT_FMT_CHECKPOINT_INTERVAL,
                 CheckpointInterval);
  }
  else
  {
    TLogger::Log(TLogger::ADVANCED,OUT_FMT_SEPARATOR);
  }


  TLogger::Log(TLogger::ADVANCED,
               OUT_FMT_COMPRESSION_LEVEL,
               CompressionLevel);

  TLogger::Log(TLogger::FULL,
               OUT_FMT_PRINT_PROGRESS_INTERVAL,
               ProgressPrintInterval);

  if (BenchmarkFlag)
  TLogger::Log(TLogger::FULL,
               OUT_FMT_BENCHMARK_TIME_STEP,
               BenchmarkTimeStepsCount);


  TLogger::Log(TLogger::ADVANCED,OUT_FMT_SAMPLING_FLAGS);


  std::string SampledQuantitiesList = "";

  // Sampled p quantities
  if (Store_p_raw)
  {
    SampledQuantitiesList += "p_raw, ";
  }
  if (Store_p_rms)
  {
    SampledQuantitiesList += "p_rms, ";
  }
  if (Store_p_max)
  {
    SampledQuantitiesList += "p_max, ";
  }
  if (Store_p_min)
  {
    SampledQuantitiesList += "p_min, ";
  }
  if (Store_p_max_all)
  {
    SampledQuantitiesList += "p_max_all, ";
  }
  if (Store_p_min_all)
  {
    SampledQuantitiesList += "p_min_all, ";
  }
  if (Store_p_final)
  {
    SampledQuantitiesList += "p_final, ";
  }

  // Sampled u quantities
  if (Store_u_raw)
  {
    SampledQuantitiesList += "u_raw, ";
  }
  if (Store_u_rms)
  {
    SampledQuantitiesList += "u_rms, ";
  }
  if (Store_u_max)
  {
    SampledQuantitiesList += "u_max, ";
  }
  if (Store_u_min)
  {
    SampledQuantitiesList += "u_min, ";
  }
  if (Store_u_max_all)
  {
    SampledQuantitiesList += "u_max_all, ";
  }
  if (Store_u_min_all)
  {
    SampledQuantitiesList += "u_min_all, ";
  }
  if (Store_u_final)
  {
    SampledQuantitiesList += "u_final, ";
  }

  if (Store_u_non_staggered_raw)
  {
    SampledQuantitiesList += "u_non_staggered_raw, ";
  }

  // remove comma and space symbols
  if (SampledQuantitiesList.length() > 0)
  {
    SampledQuantitiesList.pop_back();
    SampledQuantitiesList.pop_back();
  }

  TLogger::Log(TLogger::ADVANCED,
               TLogger::WordWrapString(SampledQuantitiesList,
                                       " ",2).c_str());

  TLogger::Log(TLogger::ADVANCED,OUT_FMT_SEPARATOR);

  TLogger::Log(TLogger::ADVANCED,
              OUT_FMT_SAMPLING_BEGINS_AT,
              StartTimeStep+1);

  if (CopySensorMask)
  {
    TLogger::Log(TLogger::ADVANCED,
                 OUT_FMT_COPY_SENSOR_MASK);
  }
}// end of PrintSetup
//------------------------------------------------------------------------------

/**
 * Parse command line.
 * @param [in, out] argc
 * @param [in, out] argv
 */
void TCommandLineParameters::ParseCommandLine(int argc, char** argv)
{
  char c;
  int longIndex = -1;
  bool CheckpointFlag = false;

  const int ErrorLineIndentation = 9;

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
          InputFileName = optarg;
        }
        else
        {
          PrintUsage();
          TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_INPUT_FILE,
                                                             " ",
                                                             ErrorLineIndentation).c_str());
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
          OutputFileName = optarg;
        }
        else
        {
          PrintUsage();
          TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_OUTPUT_FILE,
                                                             " ",
                                                             ErrorLineIndentation).c_str());
        }
        break;
      }

      case 'r':
      {
        try
        {
          int ConvertedValue = std::stoi(optarg);
          if ((ConvertedValue  < 1) || (ConvertedValue  > 100))
          {
            throw std::invalid_argument("-r");
          }
          ProgressPrintInterval = std::stoll(optarg);
        }
        catch (...)
        {
          PrintUsage();
          TLogger::ErrorAndTerminate(TLogger::WordWrapString(FMT_NO_PROGRESS_PRINT_INTERVAL,
                                                             " ",
                                                             ErrorLineIndentation).c_str());
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
          NumberOfThreads = std::stoll(optarg);
        }
        catch (...)
        {
          PrintUsage();
          TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_THREAD_NUMBER,
                                                             " ",
                                                             ErrorLineIndentation).c_str());
        }
        break;
      }

      case 'g':
      {
        try
        {
          GPUDeviceIdx = std::stoi(optarg);
          if (GPUDeviceIdx < 0)
          {
            throw std::invalid_argument("-g");
          }
        }
        catch (...)
        {
          PrintUsage();
          TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_GPU_NUMBER,
                                                             " ",
                                                             ErrorLineIndentation).c_str());
        }
        break;
      }

      case 'c':
      {
        try
        {
          int CovertedValue = std::stoi(optarg);
          if ((CovertedValue < 0) || (CovertedValue > 9))
          {
            throw std::invalid_argument("-c");
          }
          CompressionLevel = std::stoll(optarg);
        }
        catch (...)
        {
          PrintUsage();
          TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_COMPRESSION_LEVEL,
                                                             " ",
                                                             ErrorLineIndentation).c_str());
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
          StartTimeStep = std::stoll(optarg) - 1;
        }
        catch (...)
        {
          PrintUsage();
          TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_START_TIME_STEP,
                                                             " ",
                                                             ErrorLineIndentation).c_str());
        }
        break;
      }

      case 1: // benchmark
      {
        try
        {
          BenchmarkFlag = true;
          if (std::stoll(optarg) <= 0)
          {
            throw std::invalid_argument("benchmark");
          }
          BenchmarkTimeStepsCount = std::stoll(optarg);
        }
        catch (...)
        {
          PrintUsage();
          TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_BENCHMARK_STEP_SET,
                                                             " ",
                                                             ErrorLineIndentation).c_str());
        }
        break;
      }

      case 2: // copy_sensor_mask
      {
        CopySensorMask = true;
        break;
      }

      case 3: // checkpoint_file
      {
        CheckpointFlag = true;
        // test if the wile was correctly entered (if not, getopt could eat
        // the following parameter)
        if ((optarg != NULL) &&
            ((strlen(optarg) > 0) && (optarg[0] != '-')))
        {
          CheckpointFileName = optarg;
        }
        else
        {
          PrintUsage();
          TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_CHECKPOINT_FILE,
                                                             " ",
                                                             ErrorLineIndentation).c_str());
        }
        break;
      }

      case 4: // checkpoint_interval
      {
        try
        {
          CheckpointFlag = true;
          if (std::stoll(optarg) <= 0)
          {
           throw std::invalid_argument("checkpoint_interval");
          }
          CheckpointInterval = std::stoll(optarg);
        }
        catch (...)
        {
          PrintUsage();
          TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_CHECKPOINT_INTERVAL,
                                                             " ",
                                                             ErrorLineIndentation).c_str());
        }
        break;
      }

      case 5: // verbose
      {
        try
        {
          int VerboseLevel = std::stoi(optarg);
          if ((VerboseLevel < 0) || (VerboseLevel > 2))
          {
            throw std::invalid_argument("verbose");
          }
          TLogger::SetLevel(static_cast<TLogger::TLogLevel> (VerboseLevel));
        }
        catch (...)
        {
          PrintUsage();
          TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_VERBOSE_LEVEL,
                                                             " ",
                                                             ErrorLineIndentation).c_str());
        }
        break;
      }

      case 6: // version
      {
        PrintVersion = true;
        break;
      }

      case 'p':
      {
        Store_p_raw = true;
        break;
      }

      case 10: // p_rms
      {
        Store_p_rms = true;
        break;
      }

      case 11: // p_max
      {
        Store_p_max = true;
        break;
      }

      case 12: // p_min
      {
        Store_p_min = true;
        break;
      }

      case 13: // p_max_all
      {
        Store_p_max_all = true;
        break;
      }

      case 14: // p_min_all
      {
        Store_p_min_all = true;
        break;
      }

      case 15: // p_final
      {
        Store_p_final = true;
        break;
      }

      case 'u':
      {
        Store_u_raw = true;
        break;
      }

      case 20: // u_rms
      {
        Store_u_rms = true;
        break;
      }

      case 21: // u_max
      {
        Store_u_max = true;
        break;
      }

      case 22: // u_min
      {
        Store_u_min = true;
        break;
      }

      case 23: // u_max_all
      {
        Store_u_max_all = true;
        break;
      }

      case 24: // u_min_all
      {
        Store_u_min_all = true;
        break;
      }

      case 25: // u_final
      {
        Store_u_final = true;
        break;
      }

      case 26: // u_non_staggered_raw
      {
        Store_u_non_staggered_raw = true;
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
                                                               ErrorLineIndentation).c_str());
            break;
          }
          case 'o':
          {
            PrintUsage();
            TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_OUTPUT_FILE,
                                                               " ",
                                                               ErrorLineIndentation).c_str());
            break;
          }

          case 'r':
          {
            PrintUsage();
            TLogger::ErrorAndTerminate(TLogger::WordWrapString(FMT_NO_PROGRESS_PRINT_INTERVAL,
                                                               " ",
                                                               ErrorLineIndentation).c_str());
            break;
          }

          case 'c':
          {
            PrintUsage();
            TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_COMPRESSION_LEVEL,
                                                               " ",
                                                               ErrorLineIndentation).c_str());
            break;
          }

          case 't':
          {
            PrintUsage();
            TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_THREAD_NUMBER,
                                                               " ",
                                                               ErrorLineIndentation).c_str());
            break;
          }

          case 'g':
          {
            PrintUsage();
            TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_GPU_NUMBER,
                                                               " ",
                                                               ErrorLineIndentation).c_str());
            break;
          }

          case 's':
          {
            PrintUsage();
            TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_START_TIME_STEP,
                                                               " ",
                                                               ErrorLineIndentation).c_str());
            break;
          }

          case 1: // benchmark
          {
            PrintUsage();
            TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_BENCHMARK_STEP_SET,
                                                               " ",
                                                               ErrorLineIndentation).c_str());
            break;
          }

          case 3: // checkpoint_file
          {
            PrintUsage();
            TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_CHECKPOINT_FILE,
                                                               " ",
                                                               ErrorLineIndentation).c_str());
            break;
          }

          case 4: // checkpoint_interval
          {
            PrintUsage();
            TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_CHECKPOINT_INTERVAL,
                                                               " ",
                                                               ErrorLineIndentation).c_str());
            break;
          }

          case 5: // verbose
          {
            PrintUsage();
            TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_VERBOSE_LEVEL,
                                                               " ",
                                                               ErrorLineIndentation).c_str());
            break;
          }

          default :
          {
            PrintUsage();
            TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_UNKNOW_PARAMETER_OR_ARGUMENT,
                                                               " ",
                                                               ErrorLineIndentation).c_str());
            break;
          }
        }
      }

      default:
      {
        PrintUsage();
        TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_UNKNOWN_PARAMETER,
                                                           " ",
                                                           ErrorLineIndentation).c_str());
        break;
      }
    }
  }

  if (PrintVersion) return;

  //-- Post checks --//
  if (InputFileName == "")
  {
    PrintUsage();
    TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_INPUT_FILE,
                                                       " ",
                                                       ErrorLineIndentation).c_str());
  }

  if (OutputFileName == "")
  {
    PrintUsage();
    TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_OUTPUT_FILE,
                                                       " ",
                                                       ErrorLineIndentation).c_str());
  }

  if (CheckpointFlag)
  {
    if (CheckpointFileName == "")
    {
      PrintUsage();
      TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_CHECKPOINT_FILE,
                                                         " ",
                                                         ErrorLineIndentation).c_str());
    }

    if (CheckpointInterval <= 0)
    {
      PrintUsage();
      TLogger::ErrorAndTerminate(TLogger::WordWrapString(ERR_FMT_NO_CHECKPOINT_INTERVAL,
                                                         " ",
                                                         ErrorLineIndentation).c_str());
    }
  }

  // set a default flag if necessary
  if (!(Store_p_raw     || Store_p_rms     || Store_p_max   || Store_p_min ||
        Store_p_max_all || Store_p_min_all || Store_p_final ||
        Store_u_raw     || Store_u_non_staggered_raw        ||
        Store_u_rms     || Store_u_max     || Store_u_min   ||
        Store_u_max_all || Store_u_min_all || Store_u_final ))
  {
    Store_p_raw = true;
  }
}// end of ParseCommandLine
//------------------------------------------------------------------------------
