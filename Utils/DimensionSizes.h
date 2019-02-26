/**
 * @file      DimensionSizes.h
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The header file containing the structure with 3D dimension sizes.
 *
 * @version   kspaceFirstOrder3D 3.6
 *
 * @date      09 August    2011, 12:34 (created) \n
 *            26 February  2019, 16:24 (revised)
 *
 * @copyright Copyright (C) 2019 Jiri Jaros and Bradley Treeby.
 *
 * This file is part of the C++ extension of the [k-Wave Toolbox](http://www.k-wave.org).
 *
 * This file is part of the k-Wave. k-Wave is free software: you can redistribute it and/or modify it under the terms
 * of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * k-Wave is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for
 * more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along with k-Wave.
 * If not, see [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/).
 */


#ifndef DIMENSION_SIZES_H
#define DIMENSION_SIZES_H

#include <cstdlib>


#if (defined(__AVX2__)) || (defined(__AVX__))
/**
 * @var   kDataAlignment
 * @brief Memory alignment for AVX and AVX2 (32B).
 */
constexpr int kDataAlignment  = 32;
#elif ((defined(__SSE4_2__)) || (defined(__SSE4_1__)) || (defined(__SSE3__))  || (defined(__SSE2__)))
/**
 * @var   kDataAlignment
 * @brief Memory alignment for SSE2, SSE3, SSE4 (16B).
 */
constexpr int kDataAlignment  = 16;
#else
  /**
   * @var     kDataAlignment
   * @brief   Default memory alignment.
   * @details Default memory alignment is oriented on new, yet unknown, architectures with wider SIMD units, possible
   *          512b.
   */
  constexpr int kDataAlignment  = 64;
#endif

/**
 * @struct  DimensionSizes
 * @brief   Structure with 4D dimension sizes (up to 3 in space and 1 in time).
 * @details Structure containing dimension sizes. \n
 *   \li 4D objects (Nx, Ny, Nz, Nt).
 *   \li 3D objects (Nx, Ny, Nz, 0) - Nt must be 0. If we don't sample in time we make 3D datasets, otherwise 4D.
 *   \li 2D objects (Nx, Ny,  1, 0) - Nt must be 1 and Nz must be 1 since all datasets are stored internally as 3D.
 *
 * The structure contains only POD, so no C++ stuff is necessary.
 */
struct DimensionSizes
{
  /// Default constructor.
  DimensionSizes() : nx(0), ny(0), nz(0), nt(0) {};

  /**
   * @brief   Constructor.
   * @details Constructor.
   * @param [in] x, y, z, t - Three spatial dimensions and time.
   */
  DimensionSizes(size_t x, size_t y, size_t z, size_t t = 0)
    : nx(x), ny(y), nz(z), nt(t)
  {};

  /// Default constructor.
  ~DimensionSizes() = default;

  /**
   * @brief   Get the number of elements in all used dimensions.
   * @details Get the number of elements in used dimension sizes.
   * @return  the number of elements the domain holds.
   */
  inline size_t nElements() const
  {
    return (!is4D()) ? nx * ny * nz : nx * ny * nz * nt;
  };

  /**
   * @brief  Does the object only include 2 dimensions?
   * @return true if the dimensions are 2D.
   */
  inline bool is2D() const
  {
    return ((nz == 1) && (nt == 0));
  };

  /**
   * @brief  Does the object only include 3 spatial dimensions?
   * @return true if the dimensions are 3D.
   */
  inline bool is3D() const
  {
    return (nz > 1) && (nt == 0);
  };

  /**
   * @brief  Does the object include all spatial and temporal dimensions?
   * @return true if the dimensions are 4D.
   */
  inline bool is4D() const
  {
    return (nt > 0);
  };

  /**
   * @brief Operator ==
   * @param [in] other  - The second operand to compare with.
   * @return true if the dimension sizes are equal.
   */
  inline bool operator==(const DimensionSizes& other) const
  {
    return ((nx == other.nx) && (ny == other.ny) && (nz == other.nz) && (nt == other.nt));
  };

  /**
   * @brief Operator !=
   * @param [in] other - the second operand to compare with.
   * @return true if !=
   */
  inline bool operator!=(const DimensionSizes& other) const
  {
    return ((nx != other.nx) || (ny != other.ny) || (nz != other.nz) || (nt != other.nt));
  };

  /**
   * @brief Operator -
   *
   * Get the size of the cube defined by two corners.
   *
   * @param [in] op1 - Usually bottom right corner.
   * @param [in] op2 - Usually top left corner.
   * @return the size of the inner cuboid
   */
  inline friend DimensionSizes operator-(const DimensionSizes& op1,
                                         const DimensionSizes& op2)
  {
    // +1 because of planes (10.10.1 - 60.40.1)
    if (!op1.is4D() && !op2.is4D())
    {
      return DimensionSizes(op1.nx - op2.nx + 1, op1.ny - op2.ny + 1, op1.nz - op2.nz + 1);
    }
    else
    {
      return DimensionSizes(op1.nx - op2.nx + 1, op1.ny - op2.ny + 1,
                            op1.nz - op2.nz + 1, op1.nt - op2.nt + 1);
    }
  };

  /// Number of elements in the x direction.
  size_t nx;
  /// Number of elements in the y direction.
  size_t ny;
  /// Number of elements in the z direction.
  size_t nz;
  /// Number of time steps (for time series datasets).
  size_t nt;
}; // end of DimensionSizes
//----------------------------------------------------------------------------------------------------------------------

#endif	/* #define	DIMENSION_SIZES_H */
