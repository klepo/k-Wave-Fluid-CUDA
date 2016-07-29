/**
 * @file        Logger.cpp
 *
 * @author      Jiri Jaros              \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The implementation file containing a class responsible for printing out
 *              info and error messages (stdout, and stderr).
 *
 * @version     kspaceFirstOrder3D 3.4
 *
 * @date        19 April    2016, 12:52 (created) \n
 *              29 July     2016, 16:44 (revised)
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

#include <cstdarg>
#include <string>
#include <sstream>

#include <Logger/Logger.h>
#include <Logger/OutputMessages.h>
#include <Logger/ErrorMessages.h>

using std::string;

//------------------------------------------------------------------------------------------------//
//--------------------------------------- Public methods -----------------------------------------//
//------------------------------------------------------------------------------------------------//

/// static declaration of the LogLevel private field
TLogger::TLogLevel TLogger::logLevel = BASIC;


/**
 * Initialise or change logging level.
 *
 * @param [in] actualLogLevel - Log level for the logger
 */
void TLogger::SetLevel(const TLogLevel actualLogLevel)
{
  logLevel = actualLogLevel;
}/// end of SetLevel
//--------------------------------------------------------------------------------------------------


/**
 * Log desired activity.
 *
 * @param [in] queryLevel - Log level of the message
 * @param [in] message    - Message to log
 */
void TLogger::Log(const TLogLevel queryLevel,
                  const string&   message)
{
  if (queryLevel <= TLogger::logLevel)
  {
    std::cout << message;
  }
}// end of Log
//--------------------------------------------------------------------------------------------------

/**
 * Log an error.
 *
 * @param [in] errorMessage - Error message to be printed out.
 */
void TLogger::Error(const string& errorMessage)
{
  std::cerr << ERR_FMT_HEAD;
  std::cerr << errorMessage;
  std::cerr << ERR_FMT_TAIL;
}// end of Error
//--------------------------------------------------------------------------------------------------

/**
 * Log an error and terminate the execution.
 *
 * @param [in] errorMessage - error message to be printed to stderr
 */
void TLogger::ErrorAndTerminate(const string& errorMessage)
{
  std::cerr << ERR_FMT_HEAD;
  std::cerr << errorMessage;
  std::cerr << ERR_FMT_TAIL;

  exit(EXIT_FAILURE);
}// end of ErrorAndTerminate
//--------------------------------------------------------------------------------------------------

/**
 * Flush logger, output messages only.
 *
 * @param [in] queryLevel - Log level of the flush
 */
void TLogger::Flush(const TLogLevel queryLevel)
{
  if (queryLevel <= TLogger::logLevel)
  {
    std::cout.flush();
  }
}// end of Flush
//--------------------------------------------------------------------------------------------------


/**
 * Wrap the line based on delimiters and align it with the rest of the logger output.
 *
 * @param [in] inputString - Input string
 * @param [in] delimiters  - String of delimiters, every char is a delimiter
 * @param [in] indentation - Indentation from the beginning
 * @param [in] lineSize    - Line size
 * @return Wrapped string
 *
 * @note The string must not contain tabulator and end-of-line characters.
 */
string TLogger::WordWrapString(const string& inputString,
                               const string& delimiters,
                               const int     indentation,
                               const int     lineSize)
{
  std::istringstream textStream(inputString);
  string wrappedText;
  string word;
  string indentationString = OUT_FMT_VERTICAL_LINE;


  // create indentation
  for (int i = 0; i < indentation - 1; i++)
  {
    indentationString += ' ';
  }

  wrappedText += OUT_FMT_VERTICAL_LINE + " ";
  int spaceLeft = lineSize - 2;

  // until the text is empty
  while (textStream.good())
  {
    word = GetWord(textStream, delimiters);
    if (spaceLeft < (int) word.length() + 3)
    { // fill the end of the line
      for ( ; spaceLeft > 2; spaceLeft--)
      {
        wrappedText += " ";
      }
      wrappedText += " " + OUT_FMT_VERTICAL_LINE + "\n" + indentationString + word;
      spaceLeft = lineSize - (word.length() + indentation);
    }
    else
    {
      // add the word at the same line
      wrappedText += word;
      spaceLeft -= word.length();

      char c;
      if (textStream.get(c).good())
      {
        wrappedText += c;
        spaceLeft--;
      }
    }
  }

  // fill the last line
  for ( ; spaceLeft > 2; spaceLeft--)
  {
    wrappedText += " ";
  }
  wrappedText += " "+ OUT_FMT_VERTICAL_LINE + "\n";

  return wrappedText;
}// end of WordWrapFileName
//--------------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------------//
//--------------------------------------- Private methods ----------------------------------------//
//------------------------------------------------------------------------------------------------//

/**
 * Extract a word from a string stream based on delimiters.
 *
 * @param [in,out] textStream - Input text stream
 * @param [in]     delimiters - List of delimiters as a single string
 * @return         A word firm the string
 */
string TLogger::GetWord(std::istringstream& textStream,
                        const string&       delimiters)
{
  string word = "";
  char c;

  while (textStream.get(c))
  {
    if (delimiters.find(c) != string::npos)
    {
      textStream.unget();
      break;
    }
    word += c;
  }

  return word;
}// end of GetWord
//--------------------------------------------------------------------------------------------------
