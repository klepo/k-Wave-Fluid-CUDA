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
 *              18 July     2016, 13:35 (revised)
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
#include <string>
#include <sstream>

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
 * @param [in] QueryLevel - Log level of the message
 * @param [in] Format     - Format of the message
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
 * Log an error
 * @param [in] Format - Format of the message
 * @param ...
 */
void TLogger::Error(const char* Format,
                    ...)
{
  fprintf(stderr, ERR_FMT_HEAD);

  va_list args;
  va_start(args, Format);
  vfprintf(stderr, Format, args);
  va_end(args);

  fprintf(stderr, ERR_FMT_TAIL);
}// end of Error
//------------------------------------------------------------------------------

/**
 * Log an error and Terminate
 * @param [in] Format - Format of the message
 * @param ...
 */
void TLogger::ErrorAndTerminate(const char* Format,
                                 ...)
{
  fprintf(stderr, ERR_FMT_HEAD);

  va_list args;
  va_start(args, Format);
  vfprintf(stderr, Format, args);
  va_end(args);

  fprintf(stderr, ERR_FMT_TAIL);

  exit(EXIT_FAILURE);
}// end of ErrorAndTerminate
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


/**
 * Wrap the line and based on delimiters and align it with the rest of
 * the logger output
 * @param [in] InputString - input string
 * @param [in] Delimiters  - string of delimiters, every char is a delimiter
 * @param [in] Indentation - indentation from the beginning
 * @param [in] LineSize    - line size
 * @return wrapped string
 *
 * @note The string must not contain \t and \n characters
 */
std::string TLogger::WordWrapString(const std::string& InputString,
                                    const std::string& Delimiters,
                                    const int          Indentation,
                                    const int          LineSize)
{
  std::istringstream TextStream(InputString);
  std::string WrappedText;
  std::string Word;
  std::string IndentationString = OUT_FMT_VerticalLine;


  // create indentation
  for (int i = 0; i < Indentation - 1; i++)
  {
    IndentationString += ' ';
  }

  WrappedText += std::string(OUT_FMT_VerticalLine) + " ";
  int SpaceLeft = LineSize - 2;

  // until the text is empty
  while (TextStream.good())
  {
    Word = GetWord(TextStream, Delimiters);
    if (SpaceLeft < (int) Word.length() + 3)
    { // fill the end of the line
      for ( ; SpaceLeft > 2; SpaceLeft--)
      {
        WrappedText += " ";
      }
      WrappedText += " "+ std::string(OUT_FMT_VerticalLine) +"\n" + IndentationString + Word;
      SpaceLeft = LineSize - (Word.length() + Indentation);
    }
    else
    {
      // add the word at the same line
      WrappedText += Word;
      SpaceLeft -= Word.length();

      char c;
      if (TextStream.get(c).good())
      {
        WrappedText += c;
        SpaceLeft--;
      }
    }
  }

  // fill the last line
  for ( ; SpaceLeft > 2; SpaceLeft--)
  {
    WrappedText += " ";
  }
  WrappedText += " "+ std::string(OUT_FMT_VerticalLine) + "\n";

  return WrappedText;
}// end of WordWrapFileName
//------------------------------------------------------------------------------

//----------------------------------------------------------------------------//
//------------------------------    Private    -------------------------------//
//----------------------------------------------------------------------------//

/**
 * Extract a word from a string stream based on delimiters
 * @param [in,out] TextStream - Input text stream
 * @param [in]     Delimiters - list of delimiters as a single string
 * @return         a word firm the string
 */
std::string TLogger::GetWord(std::istringstream& TextStream,
                             const std::string&  Delimiters)
{
  std::string Word = "";
  char c;

  while (TextStream.get(c))
  {
    if (Delimiters.find(c) != std::string::npos)
    {
      TextStream.unget();
      break;
    }
    Word += c;
  }

  return Word;
}// end of GetWord
//------------------------------------------------------------------------------
