/**
 * @file        CommandLineParameters.cpp
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The implementation file containing the command line parameters.
 *
 * @version     kspaceFirstOrder3D 3.3
 * @date        29 August   2012, 11:25 (created) \n
 *              04 November 2014, 17:24 (revised)
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
#include "../GetoptWin64/Getopt.h"
#endif

#include <stdio.h>
#include <string.h>
#include <sstream>
#include <omp.h>

#include "./CommandLineParameters.h"
#include "../Utils/ErrorMessages.h"

//--------------------------------------------------------------------------//
//-------------------------- Constants -------------------------------------//
//--------------------------------------------------------------------------//

//--------------------------------------------------------------------------//
//--------------------------- Public   -------------------------------------//
//--------------------------------------------------------------------------//

/**
 * Constructor
 */
TCommandLineParameters::TCommandLineParameters()
{
    InputFileName = "";
    OutputFileName = "";
    CheckpointFileName = "";
    VerboseInterval = 5;
    CompressionLevel = 3;
    BenchmarkFlag = false;
    BenchmarkTimeStepsCount = 100;
    CheckpointInterval = 0;
    PrintVersion = false;
    Store_p_raw = false;
    Store_p_rms = false;
    Store_p_max = false;
    Store_p_min = false;
    Store_p_max_all = false;
    Store_p_min_all = false;
    Store_p_final = false;
    Store_u_raw = false;
    Store_u_non_staggered_raw = false;
    Store_u_rms = false;
    Store_u_max = false;
    Store_u_min = false;
    Store_u_max_all = false;
    Store_u_min_all = false;
    Store_u_final = false;
    Store_I_avg = false;
    Store_I_max = false;
    StartTimeStep = 0;
#if CUDA_VERSION || OPENCL_VERSION
    gpu_device_id = -1;
    one_d_block_size = 0;
    three_d_block_size_x = 0;
    three_d_block_size_y = 0;
    three_d_block_size_z = 0;
#endif
    NumberOfThreads  = 1;
    VerboseInterval  = DefaultVerboseInterval;
    CompressionLevel = DefaultCompressionLevel;
}// end of TCommandLineParameters
//------------------------------------------------------------------------------

/**
 * Print usage and exit.
 */
void TCommandLineParameters::PrintUsageAndExit(){

    printf("---------------------------------- Usage ---------------------------------\n");
    printf("Mandatory parameters:\n");
    printf("  -i Input_file_name              : HDF5 input file\n");
    printf("  -o Output_file_name             : HDF5 output file\n");
    printf("\n");
    printf("Optional parameters: \n");
#if CUDA_VERSION || OPENCL_VERSION
    printf("  -g GPU_device_number            : GPU device to run on\n");
    printf("                                      (default = device with most global memory)\n");
#endif
    printf("  -t Number_of_CPU_threads        : Number of CPU threads\n");
    printf("                                      (default = %d)\n",1);
    printf("  -r Progress_report_interval_in_%%: Progress print interval\n");
    printf("                                      (default = %d%%)\n",DefaultVerboseInterval);
    printf("  -c Output_file_compression_level: Deflate compression level <0,9>\n");
    printf("                                      (default = %d)\n",DefaultCompressionLevel );
    printf("  --benchmark <steps>             : Run a specified number of time steps\n");
    printf("  --checkpoint_file <file_name>   : HDF5 Checkpoint file\n");
    printf("  --checkpoint_interval <seconds> : Stop after a given number of seconds and\n");
    printf("                                      store the actual state\n");

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
    printf("  -I                              : Store intensity\n");
    printf("                                      (the same as --I_avg)\n");
    printf("  --I_avg                         : Store avg of intensity\n");
    printf("  --I_max                         : Store max of intensity\n");
    printf("\n");
#if CUDA_VERSION || OPENCL_VERSION
    printf("  --1DBlockSize i                 : Run 1D CUDA kernels with block size i\n");
    printf("                                      (default is 256 threads per block)\n");
    printf("  --3DBlockSize x,y,z             : Run 3D CUDA kernels with block size in x, y and z dimensions\n");
    printf("                                      (default is x=256 y=1 z=1 threads per block)\n");
    printf("\n");
#endif
    printf("  -s Start_time_step              : Time step when data collection begins\n");

    printf("                                      (default = 1)\n");
    printf("--------------------------------------------------------------------------\n");
    printf("\n");

    exit(EXIT_FAILURE);

}// end of PrintUsageAndExit
//------------------------------------------------------------------------------

/**
 * Print setup.
 */
void TCommandLineParameters::PrintSetup(){

    printf("List of enabled parameters:\n");

    printf("  Input  file               %s\n",InputFileName.c_str());
    printf("  Output file               %s\n",OutputFileName.c_str());
    printf("\n");
#if CUDA_VERSION || OPENCL_VERSION
    if(gpu_device_id == -1){
        printf("  GPU device number     default\n");
    }else{
        printf("  GPU device number     %d\n", gpu_device_id);
    }
#endif
    printf("  Number of threads         %d\n", NumberOfThreads);
    printf("  Verbose interval[%%]      %d\n", VerboseInterval);
    printf("  Compression level         %d\n", CompressionLevel);
    printf("\n");
    printf("  Benchmark flag            %d\n", BenchmarkFlag);
    printf("  Benchmark time steps      %d\n", BenchmarkTimeStepsCount);
    printf("\n");
    printf("  Checkpoint_file           %s\n", CheckpointFileName.c_str());
    printf("  Checkpoint_interval       %d\n", CheckpointInterval);
    printf("\n");
    printf("  Store p_raw               %d\n", Store_p_raw);
    printf("  Store p_rms               %d\n", Store_p_rms);
    printf("  Store p_max               %d\n", Store_p_max);
    printf("  Store p_min               %d\n", Store_p_min);
    printf("  Store p_max_all           %d\n", Store_p_max_all);
    printf("  Store p_min_all           %d\n", Store_p_min_all);
    printf("  Store p_final             %d\n", Store_p_final);
    printf("\n");
    printf("  Store u_raw               %d\n", Store_u_raw);
    printf("  Store u_non_staggered_raw %d\n", Store_u_non_staggered_raw);
    printf("  Store u_rms               %d\n", Store_u_rms);
    printf("  Store u_max               %d\n", Store_u_max);
    printf("  Store u_min               %d\n", Store_u_min);
    printf("  Store u_max_all           %d\n", Store_u_max_all);
    printf("  Store u_min_all           %d\n", Store_u_min_all);
    printf("  Store u_final             %d\n", Store_u_final);
    printf("\n");
    printf("  Store I_avg               %d\n", Store_I_avg);
    printf("  Store I_max               %d\n", Store_I_max);
    printf("\n");
#if CUDA_VERSION || OPENCL_VERSION
    if(one_d_block_size == 0){
        printf("  1D Block Size   =     default\n");
    }else{
        printf("  1D Block Size   =     %d\n", one_d_block_size);
    }
    if(three_d_block_size_x == 0 ||
            three_d_block_size_y == 0 ||
            three_d_block_size_z == 0){
        printf("  3D Block Size   =     default\n");
    }else{
        printf("  3D Block Size X =     %d\n", three_d_block_size_x);
        printf("                Y =     %d\n", three_d_block_size_y);
        printf("                Z =     %d\n", three_d_block_size_z);
    }
    printf("\n");
#endif
    printf("  Collection begins at  %d\n", StartTimeStep+1);

}// end of PrintSetup
//------------------------------------------------------------------------------

/**
 * Parse command line.
 * @param [in, out] argc
 * @param [in, out] argv
 */
void TCommandLineParameters::ParseCommandLine(int argc, char** argv){
    char c;
    int longIndex;
    bool CheckpointFlag = false;

    const char * shortOpts = "i:o:r:c:t:g:puIhs:";

    const struct option longOpts[] = {
        { "benchmark", required_argument , NULL, 0},
        { "help", no_argument, NULL, 'h' },
        { "version", no_argument, NULL, 0 },
        { "checkpoint_file"    , required_argument, NULL, 0 },
        { "checkpoint_interval", required_argument, NULL, 0 },

        { "p_raw", no_argument, NULL, 'p' },
        { "p_rms", no_argument, NULL, 0 },
        { "p_max", no_argument, NULL, 0 },
        { "p_min", no_argument, NULL, 0},
        { "p_max_all", no_argument, NULL, 0},
        { "p_min_all", no_argument, NULL, 0},
        { "p_final", no_argument, NULL, 0 },

        { "u_raw", no_argument, NULL, 'u' },
        { "u_non_staggered_raw", no_argument, NULL, 0},
        { "u_rms", no_argument, NULL, 0 },
        { "u_max", no_argument, NULL, 0 },
        { "u_min", no_argument, NULL, 0},
        { "u_max_all", no_argument, NULL, 0},
        { "u_min_all", no_argument, NULL, 0},
        { "u_final", no_argument, NULL, 0 },

        { "I_avg", no_argument, NULL, 'I' },
        { "I_max", no_argument, NULL, 0 },

        { "copy_sensor_mask", no_argument, NULL, 0},

#if CUDA_VERSION || OPENCL_VERSION
        { "1DBlockSize", required_argument, NULL, 0 },
        { "3DBlockSize", required_argument, NULL, 0 },
#endif
        { NULL, no_argument, NULL, 0 }
    };

    // Short parameters //
    while ((c = getopt_long (argc, argv, shortOpts, longOpts, &longIndex )) != -1){
        switch (c){
            case 'i':{
                InputFileName = optarg;
                break;
            }
            case 'o':{
                 OutputFileName = optarg;
                 break;
            }
            case 'r':{
                 if ((optarg == NULL) || (atoi(optarg) <= 0)) {
                     fprintf(stderr,
                             "%s",
                             CommandlineParameters_ERR_FMT_NoVerboseIntreval);
                     PrintUsageAndExit();
                 }else {
                     VerboseInterval = atoi(optarg);
                 }
                 break;
            }
            case 't':{
                 if ((optarg == NULL) || (atoi(optarg) <= 0)) {
                     fprintf(stderr,
                             "%s",
                             CommandlineParameters_ERR_FMT_NoThreadNumbers);
                     PrintUsageAndExit();
                 }else {
                     NumberOfThreads = atoi(optarg);
                 }
                 break;
            }
#if CUDA_VERSION || OPENCL_VERSION
            case 'g':{
                if ((optarg == NULL) || (atoi(optarg) < -1)) {
                    fprintf(stderr,
                            "%s",
                            CommandlineParameters_ERR_FMT_NoGPUNumbers);
                    PrintUsageAndExit();
                }else {
                    gpu_device_id = atoi(optarg);
                }
                break;
            }
#endif
            case 'c':{
                if ((optarg == NULL) || (atoi(optarg) < 0) || atoi(optarg) > 9) {
                    fprintf(stderr,
                            "%s",
                            CommandlineParameters_ERR_FMT_NoCompressionLevel);
                    PrintUsageAndExit();
                } else {
                    CompressionLevel = atoi(optarg);
                }
                break;
            }
            case 'p':{
                Store_p_raw = true;
                break;
            }
            case 'u':{
                Store_u_raw = true;
                break;
            }
            case 'I':{
                Store_I_avg = true;
                break;
            }
            case 'h':{
                PrintUsageAndExit();
                break;
            }
            case 's':{
                if ((optarg == NULL) || (atoi(optarg) < 1)) {
                    fprintf(stderr,
                            "%s",
                            CommandlineParameters_ERR_FMT_NoStartTimestep);
                    PrintUsageAndExit();
                }
                StartTimeStep = atoi(optarg) - 1;
                break;
            }
            case 0:{    /* long option without a short arg */
                if(strcmp("benchmark",longOpts[longIndex].name) == 0) {
                    BenchmarkFlag = true;
                    if ((optarg == NULL) || (atoi(optarg) <= 0)) {
                        fprintf(stderr,
                                "%s",
                                CommandlineParameters_ERR_FMT_NoBenchmarkTimeStepCount);
                        PrintUsageAndExit();
                    }else{
                        BenchmarkTimeStepsCount = atoi(optarg);
                    }
                    break;
                }
                if(strcmp("checkpoint_file", longOpts[longIndex].name ) == 0) {
                    CheckpointFlag = true;
                    if ((optarg == NULL)) {
                        fprintf(stderr,
                                "%s",
                                CommandlineParameters_ERR_FMT_NoCheckpointFile);
                        PrintUsageAndExit();
                    } else {
                        CheckpointFileName = optarg;
                    }
                    break;
                }
                if(strcmp("checkpoint_interval", longOpts[longIndex].name) == 0) {
                    CheckpointFlag = true;
                    if ((optarg == NULL) || (atoi(optarg) <= 0)) {
                        fprintf(stderr,
                                "%s",
                                CommandlineParameters_ERR_FMT_NoCheckpointInterval);
                        PrintUsageAndExit();
                    } else {
                        CheckpointInterval = atoi(optarg);
                    }
                    break;
                }
                if(strcmp("version", longOpts[longIndex].name) == 0) {
                    PrintVersion = true;
                    return;
                }
                if(strcmp("p_rms", longOpts[longIndex].name) == 0) {
                            Store_p_rms = true;
                            break;
                        }
                if(strcmp("p_max", longOpts[longIndex].name) == 0) {
                    Store_p_max = true;
                    break;
                }
                if (strcmp("p_min", longOpts[longIndex].name) == 0) {
                    Store_p_min = true;
                    break;
                }
                if (strcmp("p_max_all", longOpts[longIndex].name) == 0) {
                    Store_p_max_all = true;
                    break;
                }
                if (strcmp("p_min_all", longOpts[longIndex].name) == 0) {
                    Store_p_min_all = true;
                    break;
                }
                if(strcmp("p_final", longOpts[longIndex].name) == 0) {
                    Store_p_final = true;
                    break;
                }
                //-- velocity related flags
                else if (strcmp("u_non_staggered_raw", longOpts[longIndex].name) == 0) {
                    //currently not supported in CUDA version
                    fprintf(stderr,
                            "%s",
                            CommandlineParameters_ERR_FMT_NoCheckpointFile);
                    PrintUsageAndExit();

                    //Store_u_non_staggered_raw = true;
                }
                if(strcmp("u_rms", longOpts[longIndex].name) == 0) {
                    Store_u_rms = true;
                    break;
                }
                if(strcmp("u_max", longOpts[longIndex].name) == 0) {
                    Store_u_max = true;
                    break;
                }
                if (strcmp("u_min", longOpts[longIndex].name) == 0) {
                    Store_u_min = true;
                    break;
                }
                if (strcmp("u_max_all", longOpts[longIndex].name) == 0) {
                    Store_u_max_all = true;
                    break;
                }
                if (strcmp("u_min_all", longOpts[longIndex].name) == 0) {
                    Store_u_min_all = true;
                    break;
                }
                if(strcmp("u_final", longOpts[longIndex].name) == 0) {
                    Store_u_final = true;
                    break;
                }
                if(strcmp("I_max", longOpts[longIndex].name) == 0) {
                    Store_I_max = true;
                    break;
                }
                if (strcmp("copy_sensor_mask", longOpts[longIndex].name) == 0) {
                    CopySensorMask = true;
                    break;
                }
#if CUDA_VERSION || OPENCL_VERSION
                if(strcmp("1DBlockSize",longOpts[longIndex].name) == 0) {
                    if(optarg == NULL || atoi(optarg)<0){
                        PrintUsageAndExit();
                    }
                    one_d_block_size = atoi(optarg);
                    break;
                }
                if(strcmp("3DBlockSize",longOpts[longIndex].name) == 0) {
                    string x,y,z;
                    std::istringstream liness(optarg);
                    getline(liness,x,',');
                    getline(liness,y,',');
                    getline(liness,z,',');
                    if(atoi(x.c_str()) == 0 &&
                            atoi(y.c_str()) == 0 &&
                            atoi(z.c_str()) == 0){
                        PrintUsageAndExit();
                    }
                    three_d_block_size_x = atoi(x.c_str());
                    three_d_block_size_y = atoi(y.c_str());
                    three_d_block_size_z = atoi(z.c_str());
                    return;
                }
#endif
                //else
                PrintUsageAndExit();
                break;
            }
            default:{
                PrintUsageAndExit();
            }
        }
    }

    //-- Post checks --//
    if (InputFileName == "") {
        fprintf(stderr,"%s",CommandlineParameters_ERR_FMT_NoInputFile);
        PrintUsageAndExit();
    }

    if (OutputFileName == "") {
        fprintf(stderr,"%s",CommandlineParameters_ERR_FMT_NoOutputFile);
        PrintUsageAndExit();
    }

    if (CheckpointFlag) {
        if (CheckpointFileName == "") {
            fprintf(stderr, "%s", CommandlineParameters_ERR_FMT_NoCheckpointFile);
            PrintUsageAndExit();
        }
        if (CheckpointInterval <= 0) {
            fprintf(stderr, "%s", CommandlineParameters_ERR_FMT_NoCheckpointInterval);
            PrintUsageAndExit();
        }
    }

    if (!(Store_p_raw     || Store_p_rms     || Store_p_max   || Store_p_min ||
          Store_p_max_all || Store_p_min_all || Store_p_final ||
          Store_u_raw     || Store_u_rms     || Store_u_max   || Store_u_min ||
          Store_u_max_all || Store_u_min_all || Store_u_final ||
          Store_I_avg     || Store_I_max)) {
        Store_p_raw = true;
    }

}// end of ParseCommandLine
//------------------------------------------------------------------------------

