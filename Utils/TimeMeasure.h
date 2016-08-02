/**
 * @file        TimeMeasure.h
 *
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief        The header file for class with time measurement
 *
 * @version     kspaceFirstOrder3D 3.4
 *
 * @date        15 August   2012, 09:35 (created) \n
 *              25 July     2016, 10:34 (revised)
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


#ifndef TIME_MEASURE_H
#define	TIME_MEASURE_H

#include <exception>

#ifdef _OPENMP
  #include <omp.h>
#else
  // Linux build
  #ifdef __linux__
    #include <sys/time.h>
  #endif
  // Windows build
  #ifdef _WIN64
    #include <Windows.h>
    #include <time.h>
  #endif
#endif

/**
 * @class  TTimeMeasure
 * @brief  Class measuring elapsed time.
 * @brief  Class measuring elapsed time, even over multiple leg simulations.
 */
class TTimeMeasure
{
  public:

    ///Default constructor
    TTimeMeasure() :
        startTime(0.0),
        stopTime(0.0),
        cumulatedElapsedTimeOverPreviousLegs(0.0)
    { };

    /// Destructor.
    virtual ~TTimeMeasure() {};

    /**
     * @brief  Copy constructor.
     * @details Copy constructor.
     * @param [in] src - The other class to copy from
     */
    TTimeMeasure(const TTimeMeasure& src) :
        startTime(src.startTime),
        stopTime (src.stopTime),
        cumulatedElapsedTimeOverPreviousLegs(src.cumulatedElapsedTimeOverPreviousLegs)
    { };

    /**
     * @brief operator =
     * @details operator =
     * @param [in] src - source
     * @return
     */
    TTimeMeasure& operator= (const TTimeMeasure& src)
    {
      if (this != &src)
      {
        startTime = src.startTime;
        stopTime  = src.stopTime;
        cumulatedElapsedTimeOverPreviousLegs = src.cumulatedElapsedTimeOverPreviousLegs;
      }
      return *this;
    };



    /// Get start timestamp.
    inline void Start()
    {
      #ifdef _OPENMP
        startTime = omp_get_wtime();
      #else
        // Linux build
        #ifdef __linux__
          timeval ActTime;
          gettimeofday(&ActTime, NULL);
          startTime = ActTime.tv_sec + ActTime.tv_usec * 1.0e-6;
        #endif
        #ifdef _WIN64
          startTime = clock() / (double) CLOCKS_PER_SEC;
        #endif
      #endif
    };

    /// Get stop timestamp.
    inline void Stop()
    {
      #ifdef _OPENMP
        stopTime = omp_get_wtime();
      #else
        // Linux build
        #ifdef __linux__
          timeval ActTime;
          gettimeofday(&ActTime, NULL);
          stopTime = ActTime.tv_sec + ActTime.tv_usec * 1.0e-6;
        #endif
        #ifdef _WIN64
          stopTime = clock() / (double) CLOCKS_PER_SEC;
        #endif
      #endif
    };

    /**
     * @brief Get elapsed time.
     * @details Get elapsed time.
     * @return elapsed time between start timestamp and stop timestamp.
     */
    inline double GetElapsedTime() const
    {
      return stopTime - startTime;
    };

    /**
     * @brief Get cumulated elapsed time over all simulation legs.
     * @details Get cumulated elapsed time over all simulation legs.
     * @return elapsed time all (including this one) legs.
     */
    inline double GetCumulatedElapsedTimeOverAllLegs() const
    {
      return cumulatedElapsedTimeOverPreviousLegs + (stopTime - startTime);
    };

    /**
     * @brief Get time spent in previous legs
     * @return elapsed time over previous legs.
     */
    inline double GetCumulatedElapsedTimeOverPreviousLegs() const
    {
      return cumulatedElapsedTimeOverPreviousLegs;
    };

    /**
     * @brief Set elapsed time in previous legs of the simulation.
     * @details Set elapsed time in previous legs of the simulation.
     * @param [in] elapsedTime - Elapsed time
     */
    void SetCumulatedElapsedTimeOverPreviousLegs(const double elapsedTime)
    {
      cumulatedElapsedTimeOverPreviousLegs = elapsedTime;
    }

  private:
    /// Start timestamp of the interval
    double startTime;
    /// Stop timestamp of the interval
    double stopTime;
    /// Elapsed time in previous simulation legs
    double cumulatedElapsedTimeOverPreviousLegs;

};// end of TTimeMeasure
//--------------------------------------------------------------------------------------------------

#endif	/* TIME_MEASURE_H */
