/**
 * @file        TimeMeasure.h
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief        The header file for class with time measurement
 *
 * @version     kspaceFirstOrder3D 3.3
 * @date        15 August   2012, 09:35 (created) \n
 *              06 November 2014, 15:08 (revised)
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


#ifndef TIMEMEASURE_H
#define	TIMEMEASURE_H

#include <exception>

#ifdef _OPENMP
  #include <omp.h>
#else
  #include <sys/time.h>
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
        StartTime(0.0),
        StopTime(0.0),
        CumulatedElapsedTimeOverPreviousLegs(0.0)
    { };

    /**
     * @brief Copy constructor.
     * @details Copy constructor.
     * @param [in] src - the other class to copy from
     */
    TTimeMeasure(const TTimeMeasure & src) :
        StartTime(src.StartTime),
        StopTime (src.StopTime),
        CumulatedElapsedTimeOverPreviousLegs(src.CumulatedElapsedTimeOverPreviousLegs)
    { };

    /**
     * @brief operator =
     * @details operator =
     * @param [in] src - source
     * @return
     */
    TTimeMeasure& operator = (const TTimeMeasure & src)
    {
      if (this != &src)
      {
        StartTime = src.StartTime;
        StopTime  = src.StopTime;
        CumulatedElapsedTimeOverPreviousLegs = src.CumulatedElapsedTimeOverPreviousLegs;
      }
      return *this;
    };

    /// Destructor.
    virtual ~TTimeMeasure() {};

    ///Get start timestamp.
    void Start()
    {
      #ifdef _OPENMP
        StartTime = omp_get_wtime();
      #else
        timeval ActTime;
        gettimeofday(&ActTime, NULL);
        StartTime = ActTime.tv_sec + ActTime.tv_usec * 1.0e-6;
      #endif
    };

    ///Get stop timestamp.
    void Stop()
    {
      #ifdef _OPENMP
        StopTime = omp_get_wtime();
      #else
        timeval ActTime;
        gettimeofday(&ActTime, NULL);
        StopTime = ActTime.tv_sec + ActTime.tv_usec * 1.0e-6;
      #endif
    };

    /**
     * @brief Get elapsed time.
     * @details Get elapsed time.
     * @return elapsed time between start timestamp and stop timestamp.
     */
    double GetElapsedTime() const
    {
      return StopTime - StartTime;
    };

   /**
     * @brief Get cumulated elapsed time over all simulation legs.
     * @details Get cumulated elapsed time over all simulation legs.
     * @return elapsed time all (including this one) legs.
     */
    double GetCumulatedElapsedTimeOverAllLegs() const
    {
      return CumulatedElapsedTimeOverPreviousLegs + (StopTime - StartTime);
    };

    /**
     * @brief Get time spent in previous legs
     * @return Elapsed time over previous legs.
     */
    double GetCumulatedElapsedTimeOverPreviousLegs() const
    {
      return CumulatedElapsedTimeOverPreviousLegs;
    };

    /**
     * @brief Set elapsed time in previous legs of the simulation.
     * @details Set elapsed time in previous legs of the simulation.
     * @param [in] ElapsedTime
     */
    void SetCumulatedElapsedTimeOverPreviousLegs(const double ElapsedTime)
    {
      CumulatedElapsedTimeOverPreviousLegs = ElapsedTime;
    }

  private:
    /// Start timestamp of the interval
    double StartTime;
    /// Stop timestamp of the interval
    double StopTime;
    /// Elapsed time in previous simulation legs
    double CumulatedElapsedTimeOverPreviousLegs;

};// end of TTimeMeasure
//------------------------------------------------------------------------------

#endif	/* TIMEMEASURE_H */
