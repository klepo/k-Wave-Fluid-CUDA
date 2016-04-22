/**
 * @file        Logger.cpp
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The implementation file containing a class responsible for printing out
 *              info and error messages (stdout, and stderr).
 *
 * @version     kspaceFirstOrder3D 3.4
 * @date        19 April    2016, 12:52 (created) \n
 *              22 April    2016, 15:22 (revised)
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

#include <cstdarg>

#include <Logger/Logger.h>
#include <Logger/OutputMessages.h>
#include <Logger/ErrorMessages.h>


//----------------------------------------------------------------------------//
//------------------------------    Public     -------------------------------//
//----------------------------------------------------------------------------//

/// static declaration of the LogLevel private field
TLogger::TLogLevel TLogger::LogLevel = Basic;


/**
 * Initialise/change logging level
 * @param [in] ActualLogLevel
 */
void TLogger::SetLevel(const TLogLevel ActualLogLevel)
{
  LogLevel = ActualLogLevel;
}/// end of SetLevel
//------------------------------------------------------------------------------


/**
 * Log desired activity
 * @param [] QueryLevel - Log level of the message
 * @param [] Format     - Format of the message
 * @param ...           - other parameters
 */
void TLogger::Log(const TLogLevel QueryLevel,
                  const char*     Format,
                  ...)
{
  if (QueryLevel <= TLogger::LogLevel )
  {
    va_list args;
    va_start(args, Format);
    vfprintf(stdout, Format, args);
    va_end(args);
  }
}// end of Log
//------------------------------------------------------------------------------

/**
 * Flush logger
 * @param [in] QueryLevel - Log level of the flush
 */
void TLogger::Flush(const TLogLevel QueryLevel)
{
  if (QueryLevel <= TLogger::LogLevel)
  {
    fflush(stdout);
  }
}// end of Flush
//------------------------------------------------------------------------------
