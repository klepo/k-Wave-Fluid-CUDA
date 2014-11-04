/**
 * @file        CommandLineParameters.h
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file containing the command line parameters.
 *
 * @version     kspaceFirstOrder3D 3.3
 * @date        29 August   2012, 11:25 (created) \n
 *              04 November 2014, 13:45 (revised)
 *
 * @section Params Command Line Parameters
 * The  C++ code requires two mandatory parameters and accepts a few optional
 * parameters and flags.  The mandatory parameters \c -i and \c -o specify the
 * input and output file. The file names respect the path conventions for
 * particular operating system.  If any of the files is not specified, cannot
 * be found or created, an error message is shown.
 *
 * The \c -t parameter sets the number of threads used, which defaults the
 * system maximum. On CPUs with Intel Hyper-Threading (HT), performance will
 * generally be better if HT is disabled in the BIOS settings. If HT is
 * switched on, the default will be to create twice as many threads as there
 * are physical processor cores, which might slow down the code execution.
 * Therefore, if the HT is on, try specifying the number of threads manually
 * for best performance (e.g., 4 for Intel Quad-Core). We recommend
 * experimenting with this parameter to find the best configuration. Note, if
 * there are other tasks being executed on the system, it might be useful to
 * further limit the number of threads to prevent system overload.
 *
 * The \c -r parameter specifies how often the information about the simulation
 * progress is printed out. By default, the C++ code prints out the actual
 * progress  of the simulation, elapsed time and estimated time of completion
 * in the interval corresponding to 5\% of the total number of times steps.
 *
 * The \c -c parameter specifies the compression level used by the ZIP library
 * to reduce the size of the output file.  The actual compression rate is
 * highly dependant on the sensor mask shape and the  range of stored
 * quantities.  Generally, the output data is very hard to compress and higher
 * compression levels do not reduce the file size much.  The default level of
 * compression has been fixed to 3 that represents the balance between
 * compression ratio and performance degradation in most cases.
 *
 * The \c --benchmark parameter enables to override the total length of
 * simulation by setting a new number of time steps to simulate.  This is
 * particularly useful for performance evaluation and benchmarking on real
 * data. As the code performance is pretty stable, 50-100 time steps are
 * usually enough to predict the simulation duration. This can help to quickly
 * find an  ideal number of CPU threads.
 *
 * The \c -h and \c --help parameters print all the parameters of the C++ code,
 * while the \c --version parameter reports the code version and internal build
 * number.
 *
 * The following flags specify the output quantities to be recorded during the
 * simulation and stored to disk.  If the \c -p or \c --p_raw is set, a time
 * series of acoustic pressure at the grid points specified by the sensor mask
 * is recorded.  If the \c --p_rms and/or \c --p_max is set, the root mean
 * square and/or maximum values of pressure based on the sensor mask are
 * recorded over a specified time period, respectively.  Finally, if \c
 * --p_final flag is set, the actual values for the entire acoustic pressure
 *  field in the final time step of the simulation is stored (this will always
 *  include the PML, regardless of the setting for \c `PMLInside').
 *
 * The similar flags are also applicable on particle velocities (\c -u, \c
 * --u_raw, \c --u_rms, \c --u_max and \c --u_final) . In this case, a raw time
 *  series, RMS, maximum and/or the entire filed will be stored for all tree
 *  spatial components of particle velocity.
 *
 * Finally, the acoustic intensity at every grid point can be calculated. Two
 * means aggression are possible: \c -I or \c -I_avg calculate and store
 * average acoustic intensity while \c --I_max calculates the maximum acoustic
* intensity.
 *
 * Note, any combination of \c p, \c u and \c I flags is admissible. If no
 * output flag is set, a time-series for acoustic pressure is stored.  If it is
 * not necessary to collect the output quantities over the entire simulation,
 * the starting time step when the collection begins can be specified by \c -s
 * parameter. Note, this parameter uses MATLAB convention (starts from 1).
 *
 *
 \verbatim ---------------------------------- Usage
 --------------------------------- Mandatory parameters: -i Input_file_name
 : HDF5 input file -o Output_file_name             : HDF5 output file

 Optional parameters:
 -t Number_of_CPU_threads        : Number of CPU threads
 (default = 1)
 -r Progress_report_interval_in_%: Progress print interval
 (default = 5%)
 -c Output_file_compression_level: Deflate compression level <0,9>
 (default = 3)
 --benchmark Time_steps          : Run a specified number of time steps

 -h                              : Print help
--help                          : Print help
--version                       : Print version

Output flags:
    -p                              : Store acoustic pressure
    (default if nothing else is on)
    (the same as --p_raw)
--p_raw                         : Store raw time series of p (default)
    --p_rms                         : Store rms of p
    --p_max                         : Store max of p
    --p_final                       : Store final pressure field

    -u                              : Store ux, uy, uz
(the same as --u_raw)
    --u_raw                         : Store raw time series of ux, uy, uz
    --u_rms                         : Store rms of ux, uy, uz
    --u_max                         : Store max of ux, uy, uz
    --u_final                       : Store final acoustic velocity

    -I                              : Store intensity
(the same as --I_avg)
    --I_avg                         : Store avg of intensity
    --I_max                         : Store max of intensity

    -s Start_time_step              : Time step when data collection begins
(default = 1)
    --------------------------------------------------------------------------
    \endverbatim
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

#ifndef TCOMMANDLINESPARAMETERS_H
#define TCOMMANDLINESPARAMETERS_H

#include <cstdlib>
#include <string>

using std::string;

    /**
     * @class TCommandLineParameters
     * @brief The class to parse and store command line parameters
     */
    class TCommandLineParameters {
        public:

            /// Constructor
            TCommandLineParameters();
            /// Destructor
            virtual ~TCommandLineParameters() {};

            /// Get input file name
            string GetInputFileName()      const {return InputFileName;};
            /// Get output file name
            string GetOutputFileName()     const {return OutputFileName;};
            /// Get Checkpoint file name
            std::string GetCheckpointFileName() const {return CheckpointFileName;};


            /// Get GPU to use
            int GetGPUDeviceID()                const {return gpu_device_id;};
            int Get1DBlockSize()                const {return one_d_block_size;};
            int Get3DBlockSize_X()               const {return three_d_block_size_x;};
            int Get3DBlockSize_Y()               const {return three_d_block_size_y;};
            int Get3DBlockSize_Z()               const {return three_d_block_size_z;};

            /// Is --benchmark flag set?
            bool IsBenchmarkFlag()              const {return BenchmarkFlag;};
            /// Is --version flag set?
            bool IsVersion()                    const {return PrintVersion; };
            /// Get benchmark time step count
            int  GetBenchmarkTimeStepsCount()   const {return BenchmarkTimeStepsCount;};

            /// Get compression level
            int  GetCompressionLevel()          const {return CompressionLevel;};
            /// Get number of threads
            int  GetNumberOfThreads()           const {return NumberOfThreads;};
            /// Get verbose interval
            int  GetVerboseInterval()           const {return VerboseInterval;};
            /// Get start time index when sensor data collection begins
            int GetStartTimeIndex()             const {return StartTimeStep;};

            /// Is checkpoint enabled?
            bool IsCheckpointEnabled()          const {return (CheckpointInterval > 0); };
            /// Get checkpoint interval
            int  GetCheckpointInterval()        const {return CheckpointInterval; };

            /// Is --p_raw set?
            bool IsStore_p_raw()                const {return Store_p_raw;};
            /// Is --p_rms set?
            bool IsStore_p_rms()                const {return Store_p_rms;};
            /// Is --p_max set?
            bool IsStore_p_max()                const {return Store_p_max;};
            /// Is --p_min set?
            bool IsStore_p_min()                const {return Store_p_min;};
            /// Is --p_max_all set?
            bool IsStore_p_max_all()            const {return Store_p_max_all;};
            /// Is --p_min_all set?
            bool IsStore_p_min_all()            const {return Store_p_min_all;};
            /// Is --p_final set?
            bool IsStore_p_final()              const {return Store_p_final;};

            /// Is --u_raw set?
            bool IsStore_u_raw()                const {return Store_u_raw;};
            /// Is --u_non_staggered_raw set?
            bool IsStore_u_non_staggered_raw()  const {return Store_u_non_staggered_raw;};
            /// Is --u_rms set?
            bool IsStore_u_rms()                const {return Store_u_rms;};
            /// Is --u_max set?
            bool IsStore_u_max()                const {return Store_u_max;};
            /// Is --u_min set?
            bool IsStore_u_min()                const {return Store_u_min;};
            /// Is --u_max_all set?
            bool IsStore_u_max_all()            const {return Store_u_max_all;};
            /// Is --u_min set?
            bool IsStore_u_min_all()            const {return Store_u_min_all;};
            /// Is --u_final set?
            bool IsStore_u_final()              const {return Store_u_final;};

            /// Is --I_avg set?
            bool IsStore_I_avg()                const {return Store_I_avg;};
            /// Is --I_max set?
            bool IsStore_I_max()                const {return Store_I_max;};

            /// is --copy_mask set
            bool IsCopySensorMask()             const {return CopySensorMask;};

            /// Print usage and exit
            void PrintUsageAndExit();
            /// Print setup
            void PrintSetup();
            /// Parse command line
            void ParseCommandLine(int argc, char** argv);

        protected:
            /// Copy constructor
            TCommandLineParameters(const TCommandLineParameters& orig);

        private:
            // Input file name
            string InputFileName;
            // Output file name
            string OutputFileName;
            string CheckpointFileName;

            // NumberOfThreads value
            int NumberOfThreads;


            // GPUDeviceID value
            int gpu_device_id;
            int one_d_block_size;
            int three_d_block_size_x;
            int three_d_block_size_y;
            int three_d_block_size_z;

            // VerboseInterval value
            int VerboseInterval;
            // CompressionLevel value
            int CompressionLevel;

            // BenchmarkFlag value
            bool BenchmarkFlag;
            // BenchmarkTimeStepsCount value
            int BenchmarkTimeStepsCount;
            /// Checkpoint interval in seconds
            int CheckpointInterval;

            // PrintVersion value
            bool PrintVersion;

            // Store_p_raw value
            bool Store_p_raw;
            // Store_p_rms value
            bool Store_p_rms;
            // Store_p_max value
            bool Store_p_max;
            /// Store_p_min value
            bool Store_p_min;
            /// Store_p_max_all value
            bool Store_p_max_all;
            /// Store_p_min_all value
            bool Store_p_min_all;
            // Store_p_final value
            bool Store_p_final;

            // Store_u_raw value
            bool Store_u_raw;
            // Store_u_non_staggered_raw value
            bool Store_u_non_staggered_raw;
            // Store_u_rms value
            bool Store_u_rms;
            // Store_u_max value
            bool Store_u_max;
            /// Store_u_min value
            bool Store_u_min;
            /// Store_u_max_all value
            bool Store_u_max_all;
            /// Store_u_min_all value
            bool Store_u_min_all;
            // Store_u_final value
            bool Store_u_final;

            // Store_I_avg value
            bool Store_I_avg;
            // Store_I_max value
            bool Store_I_max;
            /// Copy sensor mask to the output file
            bool CopySensorMask;
            // StartTimeStep value
            int StartTimeStep;

            /// Default compression level
            static const int DefaultCompressionLevel = 3;
            /// Default verbose interval
            static const int DefaultVerboseInterval  = 5;

    };// end of class TCommandLineParameters

#endif /* TCOMMANDLINESPARAMETERS_H */

