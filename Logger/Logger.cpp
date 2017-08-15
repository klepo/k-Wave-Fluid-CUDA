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
 *              17 July     2017, 15:21 (revised)
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

#include <string>
#include <sstream>

#include <Logger/Logger.h>

using std::string;

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/// static declaration of the LogLevel private field
Logger::LogLevel Logger::slogLevel = LogLevel::kBasic;


/**
 * Initialise or change logging level.
 */
void Logger::setLevel(const LogLevel actualLogLevel)
{
  slogLevel = actualLogLevel;
}// end of setLevel
//----------------------------------------------------------------------------------------------------------------------


/**
 * Log desired activity.
 */
void Logger::log(const LogLevel queryLevel,
                 const string&  message)
{
  if (queryLevel <= Logger::slogLevel)
  {
    std::cout << message;
  }
}// end of log
//----------------------------------------------------------------------------------------------------------------------

/**
 * Log an error.
 */
void Logger::error(const string& errorMessage)
{
  std::cerr << kErrFmtHead;
  std::cerr << errorMessage;
  std::cerr << kErrFmtTail;
}// end of error
//----------------------------------------------------------------------------------------------------------------------

/**
 * Log an error and terminate the execution.
 */
void Logger::errorAndTerminate(const string& errorMessage)
{
  std::cerr << kErrFmtHead;
  std::cerr << errorMessage;
  std::cerr << kErrFmtTail;

  exit(EXIT_FAILURE);
}// end of errorAndTerminate
//----------------------------------------------------------------------------------------------------------------------

/**
 * Flush logger, output messages only.
 */
void Logger::flush(const LogLevel queryLevel)
{
  if (queryLevel <= Logger::slogLevel)
  {
    std::cout.flush();
  }
}// end of flush
//----------------------------------------------------------------------------------------------------------------------


/**
 * Wrap the line based on delimiters and align it with the rest of the logger output.
 *
 */
string Logger::wordWrapString(const string& inputString,
                              const string& delimiters,
                              const int     indentation,
                              const int     lineSize)
{
  std::istringstream textStream(inputString);
  string wrappedText;
  string word;
  string indentationString = kOutFmtVerticalLine;


  // create indentation
  for (int i = 0; i < indentation - 1; i++)
  {
    indentationString += ' ';
  }

  wrappedText += kOutFmtVerticalLine + " ";
  int spaceLeft = lineSize - 2;

  // until the text is empty
  while (textStream.good())
  {
    word = getWord(textStream, delimiters);
    if (spaceLeft < static_cast<int>(word.length()) + 3)
    { // fill the end of the line
      for ( ; spaceLeft > 2; spaceLeft--)
      {
        wrappedText += " ";
      }
      wrappedText += " " + kOutFmtVerticalLine + "\n" + indentationString + word;
      spaceLeft = lineSize - (static_cast<int>(word.length()) + indentation);
    }
    else
    {
      // add the word at the same line
      wrappedText += word;
      spaceLeft -= static_cast<int>(word.length());

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
  wrappedText += " "+ kOutFmtVerticalLine + "\n";

  return wrappedText;
}// end of wordWrapFileName
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Private methods --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Extract a word from a string stream based on delimiters.
 */
string Logger::getWord(std::istringstream& textStream,
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
}// end of getWord
//----------------------------------------------------------------------------------------------------------------------
