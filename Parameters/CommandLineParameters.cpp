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
 *              03 May      2016, 13:51 (revised)
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

#include <stdio.h>
#include <string.h>
#include <sstream>
#include <limits>
#include <stdexcept>

#ifdef _OPENMP
  #include <omp.h>
#endif

#include <Parameters/CommandLineParameters.h>
#include <Logger/ErrorMessages.h>

#include <Logger/Logger.h>

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

/** Routine to word wrap filename based on / character
 * @param [in]  Text
 * @param [in]  Indentation - indentation of the second and following lines
 * @param [in ] LineLenght
 * @return wrapped string
 */
std::string TCommandLineParameters::WordWrapFileName(const std::string Text,
                                                     const size_t Indentation,
                                                     const size_t LineLength)
{
  std::istringstream words(Text);
  std::ostringstream wrapped;
  std::string word;
  std::string IndentationString;

  // create indentation
  for (size_t i = 0; i < Indentation; i++)
  {
    IndentationString += ' ';
  }


  if (words.good())
  {
    std::getline(words, word, '/');
    wrapped << word;
    size_t space_left = LineLength - word.length();

    while (words.good())
    {
      getline(words, word, '/');
      if (space_left < word.length() + 1)
      {
        wrapped << "/\n" << IndentationString << word;
        space_left = LineLength - word.length() - Indentation - 1;
      }
      else
      {
        wrapped << '/' << word;
        space_left -= word.length() + 1;
      }
    }
  }
  return wrapped.str();
}// end of WordWrapFileName
//------------------------------------------------------------------------------

/**
 * Print usage and exit.
 */
void TCommandLineParameters::PrintUsageAndExit()
{
  printf("---------------------------------- Usage ---------------------------------\n");
  printf("Mandatory parameters:\n");
  printf("  -i Input_file_name              : HDF5 input file\n");
  printf("  -o Output_file_name             : HDF5 output file\n");
  printf("\n");
  printf("Optional parameters: \n");

  #ifdef _OPENMP
    printf("  -t <num_threads>                : Number of CPU threads\n");
    printf("                                      (default = %d)\n", omp_get_num_procs());
  #endif

  printf("  -g <device_number>              : GPU device to run on\n");
  printf("                                      (default = first free)\n");

  printf("  -r Progress_report_interval_in_%%: Progress print interval\n");
  printf("                                      (default = %ld%%)\n",DefaultProgressPrintInterval);
  printf("  -c Output_file_compression_level: Deflate compression level <0,9>\n");
  printf("                                      (default = %ld)\n",DefaultCompressionLevel );
  printf("  --benchmark <steps>             : Run a specified number of time steps\n");
  printf("  --checkpoint_file <file_name>   : HDF5 Checkpoint file\n");
  printf("  --checkpoint_interval <seconds> : Stop after a given number of seconds and\n");
  printf("                                      store the actual state\n");
  printf("\n");
  printf("  --verbose <level>               : Level of verbosity <0,2>\n");
  printf("                                      0 - Basic, 1 - Advanced, 2 - Full \n");
  printf("                                     (default = Basic) \n");
  printf("\n");

  printf("  -h                              : Print help\n");
  printf("  --help                          : Print help\n");
  printf("  --version                       : Print version\n");
  printf("\n");

  printf("Output flags:\n");
  printf("  -p                              : Store acoustic pressure \n");
  printf("                                      (default if nothing else is on)\n");
  printf("                                      (the same as --p_raw)\n");
  printf("  --p_raw                         : Store raw time series of p (default)\n");
  printf("  --p_rms                         : Store rms of p\n");
  printf("  --p_max                         : Store max of p\n");
  printf("  --p_min                         : Store min of p\n");
  printf("  --p_max_all                     : Store max of p (whole domain)\n");
  printf("  --p_min_all                     : Store min of p (whole domain)\n");
  printf("  --p_final                       : Store final pressure field \n");
  printf("\n");
  printf("  -u                              : Store ux, uy, uz\n");
  printf("                                      (the same as --u_raw)\n");
  printf("  --u_raw                         : Store raw time series of ux, uy, uz\n");
  printf("  --u_non_staggered_raw           : Store non-staggered raw time series of\n");
  printf("                                      ux, uy, uz \n");
  printf("  --u_rms                         : Store rms of ux, uy, uz\n");
  printf("  --u_max                         : Store max of ux, uy, uz\n");
  printf("  --u_min                         : Store min of ux, uy, uz\n");
  printf("  --u_max_all                     : Store max of ux, uy, uz (whole domain)\n");
  printf("  --u_min_all                     : Store min of ux, uy, uz (whole domain)\n");
  printf("  --u_final                       : Store final acoustic velocity\n");
  printf("\n");

  printf("  -s Start_time_step              : Time step when data collection begins\n");
  printf("                                      (default = 1)\n");
  printf("--------------------------------------------------------------------------\n");
  printf("\n");

  exit(EXIT_FAILURE);
}// end of PrintUsageAndExit
//------------------------------------------------------------------------------

/**
 * Print out commandline parameters
 */
void TCommandLineParameters::PrintComandlineParamers()
{
  TLogger::Log(TLogger::Advanced,Main_OUT_FMT_SmallSeparator);

  TLogger::Log(TLogger::Advanced,
               WordWrapFileName(TCommandlineParamereres_OUT_FMT_InputFile + InputFileName + "\n", 14, 40).c_str());

  TLogger::Log(TLogger::Advanced,
               WordWrapFileName(TCommandlineParamereres_OUT_FMT_OutputFile + OutputFileName + "\n", 14, 40).c_str());


  if (IsCheckpointEnabled())
  {
    TLogger::Log(TLogger::Advanced,
               WordWrapFileName(TCommandlineParamereres_OUT_FMT_CheckpointFile + CheckpointFileName + "\n", 14, 40).c_str());

    TLogger::Log(TLogger::Advanced,Main_OUT_FMT_SmallSeparator);

    TLogger::Log(TLogger::Advanced,
                 TCommandlineParamereres_OUT_FMT_CheckpointInterval,
                 CheckpointInterval);
  }
  else
  {
    TLogger::Log(TLogger::Advanced,Main_OUT_FMT_SmallSeparator);
  }


  TLogger::Log(TLogger::Advanced,
               TCommandlineParamereres_OUT_FMT_CompressionLevel,
               CompressionLevel);

  TLogger::Log(TLogger::Full,
               TCommandlineParamereres_OUT_FMT_PrintProgressInterval,
               ProgressPrintInterval);

  if (BenchmarkFlag)
  TLogger::Log(TLogger::Full,
               TCommandlineParamereres_OUT_FMT_BenchmarkTimeStepCount,
               BenchmarkTimeStepsCount);


  TLogger::Log(TLogger::Advanced,Main_OUT_FMT_SmallSeparator);


  constexpr int PaddingOffset = 40 - 18;
  // Sampled p quantities
  if (Store_p_raw)
  {
    const int paddingLength = PaddingOffset - strlen("p_raw");
    TLogger::Log(TLogger::Advanced,
                 TCommandlineParamereres_OUT_FMT_QuantitySampling,
                 paddingLength,
                 TKSpaceFirstOrder3DSolver_OUT_FMT_CUDADeviceNamePadding,
                 "p_raw");
  }
  if (Store_p_rms)
  {
    const int paddingLength = PaddingOffset - strlen("p_rms");
    TLogger::Log(TLogger::Advanced,
                 TCommandlineParamereres_OUT_FMT_QuantitySampling,
                 paddingLength,
                 TKSpaceFirstOrder3DSolver_OUT_FMT_CUDADeviceNamePadding,
                 "p_rms");
  }
  if (Store_p_max)
  {
    const int paddingLength = PaddingOffset - strlen("p_max");
    TLogger::Log(TLogger::Advanced,
                 TCommandlineParamereres_OUT_FMT_QuantitySampling,
                 paddingLength,
                 TKSpaceFirstOrder3DSolver_OUT_FMT_CUDADeviceNamePadding,
                 "p_max");
  }
  if (Store_p_min)
  {
    const int paddingLength = PaddingOffset - strlen("p_min");
    TLogger::Log(TLogger::Advanced,
                 TCommandlineParamereres_OUT_FMT_QuantitySampling,
                 paddingLength,
                 TKSpaceFirstOrder3DSolver_OUT_FMT_CUDADeviceNamePadding,
                 "p_min");
  }
  if (Store_p_max_all)
  {
    const int paddingLength = PaddingOffset - strlen("p_max_all");
    TLogger::Log(TLogger::Advanced,
                 TCommandlineParamereres_OUT_FMT_QuantitySampling,
                 paddingLength,
                 TKSpaceFirstOrder3DSolver_OUT_FMT_CUDADeviceNamePadding,
                 "p_max_all");
  }
  if (Store_p_min_all)
  {
    const int paddingLength = PaddingOffset - strlen("p_min_all");
    TLogger::Log(TLogger::Advanced,
                 TCommandlineParamereres_OUT_FMT_QuantitySampling,
                 paddingLength,
                 TKSpaceFirstOrder3DSolver_OUT_FMT_CUDADeviceNamePadding,
                 "p_min_all");
  }
  if (Store_p_final)
  {
    const int paddingLength = PaddingOffset - strlen("p_final");
    TLogger::Log(TLogger::Advanced,
                 TCommandlineParamereres_OUT_FMT_QuantitySampling,
                 paddingLength,
                 TKSpaceFirstOrder3DSolver_OUT_FMT_CUDADeviceNamePadding,
                 "p_final");
  }

  // Sampled u quantities
  if (Store_u_raw)
  {
    const int paddingLength = PaddingOffset - strlen("u_raw");
    TLogger::Log(TLogger::Advanced,
                 TCommandlineParamereres_OUT_FMT_QuantitySampling,
                 paddingLength,
                 TKSpaceFirstOrder3DSolver_OUT_FMT_CUDADeviceNamePadding,
                 "u_raw");
  }
  if (Store_u_non_staggered_raw)
  {
    const int paddingLength = PaddingOffset - strlen("u_non_staggered_raw");
    TLogger::Log(TLogger::Advanced,
                 TCommandlineParamereres_OUT_FMT_QuantitySampling,
                 paddingLength,
                 TKSpaceFirstOrder3DSolver_OUT_FMT_CUDADeviceNamePadding,
                 "u_non_staggered_raw");
  }
  if (Store_u_rms)
  {
    const int paddingLength = PaddingOffset - strlen("u_rms");
    TLogger::Log(TLogger::Advanced,
                 TCommandlineParamereres_OUT_FMT_QuantitySampling,
                 paddingLength,
                 TKSpaceFirstOrder3DSolver_OUT_FMT_CUDADeviceNamePadding,
                 "u_rms");
  }
  if (Store_u_max)
  {
    const int paddingLength = PaddingOffset - strlen("u_max");
    TLogger::Log(TLogger::Advanced,
                 TCommandlineParamereres_OUT_FMT_QuantitySampling,
                 paddingLength,
                 TKSpaceFirstOrder3DSolver_OUT_FMT_CUDADeviceNamePadding,
                 "u_max");
  }
  if (Store_u_min)
  {
    const int paddingLength = PaddingOffset - strlen("u_min");
    TLogger::Log(TLogger::Advanced,
                 TCommandlineParamereres_OUT_FMT_QuantitySampling,
                 paddingLength,
                 TKSpaceFirstOrder3DSolver_OUT_FMT_CUDADeviceNamePadding,
                 "u_min");
  }
  if (Store_u_max_all)
  {
    const int paddingLength = PaddingOffset - strlen("u_max_all");
    TLogger::Log(TLogger::Advanced,
                 TCommandlineParamereres_OUT_FMT_QuantitySampling,
                 paddingLength,
                 TKSpaceFirstOrder3DSolver_OUT_FMT_CUDADeviceNamePadding,
                 "u_max_all");
  }
  if (Store_u_min_all)
  {
    const int paddingLength = PaddingOffset - strlen("u_min_all");
    TLogger::Log(TLogger::Advanced,
                 TCommandlineParamereres_OUT_FMT_QuantitySampling,
                 paddingLength,
                 TKSpaceFirstOrder3DSolver_OUT_FMT_CUDADeviceNamePadding,
                 "u_min_all");
  }
  if (Store_u_final)
  {
    const int paddingLength = PaddingOffset - strlen("u_final");
    TLogger::Log(TLogger::Advanced,
                 TCommandlineParamereres_OUT_FMT_QuantitySampling,
                 paddingLength,
                 TKSpaceFirstOrder3DSolver_OUT_FMT_CUDADeviceNamePadding,
                 "u_final");
  }

  TLogger::Log(TLogger::Advanced,
              TCommandlineParamereres_OUT_FMT_CollectionBeginsAt,
              StartTimeStep+1);

  if (CopySensorMask)
  {
    TLogger::Log(TLogger::Advanced,
                 TCommandlineParamereres_OUT_FMT_CopySensorMaskYes);
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
  int longIndex;
  bool CheckpointFlag = false;

  #ifdef _OPENMP
    const char * shortOpts = "i:o:r:c:t:g:puhs:";
  #else
    const char * shortOpts = "i:o:r:c:g:puIhs:";
  #endif

  const struct option longOpts[] =
  {
    { "benchmark",            required_argument, NULL, 0},
    { "help",                 no_argument, NULL,      'h'},
    { "version",              no_argument, NULL, 0 },
    { "checkpoint_file"    ,  required_argument, NULL, 0 },
    { "checkpoint_interval",  required_argument, NULL, 0 },
    { "verbose",              required_argument, NULL, 0 },

    { "p_raw",                no_argument, NULL,'p' },
    { "p_rms",                no_argument, NULL, 0 },
    { "p_max",                no_argument, NULL, 0 },
    { "p_min",                no_argument, NULL, 0},
    { "p_max_all",            no_argument, NULL, 0},
    { "p_min_all",            no_argument, NULL, 0},
    { "p_final",              no_argument, NULL, 0 },

    { "u_raw",                no_argument, NULL,'u' },
    { "u_non_staggered_raw",  no_argument, NULL, 0},
    { "u_rms",                no_argument, NULL, 0},
    { "u_max",                no_argument, NULL, 0},
    { "u_min",                no_argument, NULL, 0},
    { "u_max_all",            no_argument, NULL, 0},
    { "u_min_all",            no_argument, NULL, 0},
    { "u_final",              no_argument, NULL, 0},

    { "copy_sensor_mask",     no_argument, NULL, 0},
    { NULL,                   no_argument, NULL, 0}
  };

  // Short parameters //
  while ((c = getopt_long (argc, argv, shortOpts, longOpts, &longIndex )) != -1)
  {
    switch (c)
    {
      case 'i':
      {
        InputFileName = optarg;
        break;
      }

      case 'o':
      {
        OutputFileName = optarg;
        break;
      }

      case 'r':
      {
        try
        {
          ProgressPrintInterval = std::stoi(optarg);
          if ((ProgressPrintInterval < 1) || (ProgressPrintInterval > 100))
          {
            throw std::invalid_argument("verbose");
          }
        }
        catch (const std::invalid_argument& ia)
        {
          fprintf(stderr, CommandlineParameters_ERR_FMT_NoProgressPrintIntreval);
          PrintUsageAndExit();
        }
        break;
      }

      case 't':
      {
        try
        {
          NumberOfThreads = std::stoi(optarg);
          if (NumberOfThreads < 1)
          {
            throw std::invalid_argument("-t");
          }
        }
        catch (const std::invalid_argument& ia)
        {
          fprintf(stderr, CommandlineParameters_ERR_FMT_NoThreadNumbers);
          PrintUsageAndExit();
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
        catch (const std::invalid_argument& ia)
        {
          fprintf(stderr, CommandlineParameters_ERR_FMT_NoGPUNumbers);
          PrintUsageAndExit();
        }
        break;
      }

      case 'c':
      {
        try
        {
          CompressionLevel = std::stoi(optarg);
          if ((CompressionLevel < 0) || (CompressionLevel > 9))
          {
            throw std::invalid_argument("-c");
          }
        }
        catch (const std::invalid_argument& ia)
        {
          fprintf(stderr, CommandlineParameters_ERR_FMT_NoCompressionLevel);
          PrintUsageAndExit();
        }
        break;
      }

      case 'p':
      {
        Store_p_raw = true;
        break;
      }

      case 'u':
      {
        Store_u_raw = true;
        break;
      }

      case 'h':
      {
        PrintUsageAndExit();
        break;
      }

      case 's':
      {
        try
        {
          StartTimeStep = std::stoi(optarg) - 1;
          if (StartTimeStep < 0)
          {
            throw std::invalid_argument("-s");
          }
        }
        catch (const std::invalid_argument& ia)
        {
          fprintf(stderr, CommandlineParameters_ERR_FMT_NoStartTimestep);
          PrintUsageAndExit();
        }
        break;
      }

      /* long option without a short arg */
      case 0:
      {
        if(strcmp("benchmark",longOpts[longIndex].name) == 0)
        {
          try
          {
            BenchmarkFlag = true;
            BenchmarkTimeStepsCount = std::stoi(optarg);
            if (BenchmarkTimeStepsCount <= 0)
            {
              throw std::invalid_argument("benchmark");
            }
          }
          catch (const std::invalid_argument& ia)
          {
            fprintf(stderr, CommandlineParameters_ERR_FMT_NoBenchmarkTimeStepCount);
            PrintUsageAndExit();
          }

          break;
        }


        if(strcmp("checkpoint_file", longOpts[longIndex].name ) == 0)
        {
          CheckpointFlag = true;
          if ((optarg == NULL))
          {
            fprintf(stderr, CommandlineParameters_ERR_FMT_NoCheckpointFile);
            PrintUsageAndExit();
          }
          else
          {
            CheckpointFileName = optarg;
          }
          break;
        }

        if(strcmp("checkpoint_interval", longOpts[longIndex].name) == 0)
        {
          try
          {
            CheckpointFlag = true;
            CheckpointInterval = std::stoi(optarg);
            if (CheckpointInterval <= 0)
            {
             throw std::invalid_argument("checkpoint_interval");
            }
          }
          catch (const std::invalid_argument& ia)
          {
            fprintf(stderr, CommandlineParameters_ERR_FMT_NoCheckpointInterval);
            PrintUsageAndExit();
          }
          break;
        }

        if(strcmp("verbose", longOpts[longIndex].name) == 0)
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
          catch (const std::invalid_argument& ia)
          {
            fprintf(stderr, CommandlineParameters_ERR_FMT_BadVerboseLevel);
            PrintUsageAndExit();
          }
          break;
        }

        if(strcmp("version", longOpts[longIndex].name) == 0)
        {
          PrintVersion = true;
          break;
        }

        if(strcmp("p_rms", longOpts[longIndex].name) == 0)
        {
          Store_p_rms = true;
          break;
        }

        if(strcmp("p_max", longOpts[longIndex].name) == 0)
        {
          Store_p_max = true;
          break;
        }

        if (strcmp("p_min", longOpts[longIndex].name) == 0)
        {
          Store_p_min = true;
          break;
        }

        if (strcmp("p_max_all", longOpts[longIndex].name) == 0)
        {
          Store_p_max_all = true;
          break;
        }

        if (strcmp("p_min_all", longOpts[longIndex].name) == 0)
        {
          Store_p_min_all = true;
          break;
        }

        if(strcmp("p_final", longOpts[longIndex].name) == 0)
        {
          Store_p_final = true;
          break;
        }

        //-- velocity related flags
        else if (strcmp("u_non_staggered_raw", longOpts[longIndex].name) == 0)
        {
          Store_u_non_staggered_raw = true;
          break;
        }

        if(strcmp("u_rms", longOpts[longIndex].name) == 0)
        {
          Store_u_rms = true;
          break;
        }

        if(strcmp("u_max", longOpts[longIndex].name) == 0)
        {
          Store_u_max = true;
          break;
        }

        if (strcmp("u_min", longOpts[longIndex].name) == 0)
        {
          Store_u_min = true;
          break;
        }

        if (strcmp("u_max_all", longOpts[longIndex].name) == 0)
        {
          Store_u_max_all = true;
          break;
        }

        if (strcmp("u_min_all", longOpts[longIndex].name) == 0)
        {
          Store_u_min_all = true;
          break;
        }

        if(strcmp("u_final", longOpts[longIndex].name) == 0)
        {
          Store_u_final = true;
          break;
        }

        if (strcmp("copy_sensor_mask", longOpts[longIndex].name) == 0)
        {
          CopySensorMask = true;
          break;
        }
        //else
        PrintUsageAndExit();
        break;
      }

      default:
      {
        PrintUsageAndExit();
      }
    }
  }

  if (PrintVersion) return;

  //-- Post checks --//
  if (InputFileName == "")
  {
    fprintf(stderr, CommandlineParameters_ERR_FMT_NoInputFile);
    PrintUsageAndExit();
  }

  if (OutputFileName == "")
  {
    fprintf(stderr, CommandlineParameters_ERR_FMT_NoOutputFile);
    PrintUsageAndExit();
  }

  if (CheckpointFlag)
  {
    if (CheckpointFileName == "")
    {
      fprintf(stderr, CommandlineParameters_ERR_FMT_NoCheckpointFile);
      PrintUsageAndExit();
    }
    if (CheckpointInterval <= 0)
    {
      fprintf(stderr, CommandlineParameters_ERR_FMT_NoCheckpointInterval);
      PrintUsageAndExit();
    }
  }

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
