/*
 * jfdctfst-helium.c - fast integer FDCT (Arm Helium)
 *
 * Copyright (C) 2020, Arm Limited.  All Rights Reserved.
 * Copyright (C) 2023, Alif Semiconductor.  All Rights Reserved.
 *
 * This software is provided 'as-is', without any express or implied
 * warranty.  In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgment in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 */

#define JPEG_INTERNALS
#include "../../jinclude.h"
#include "../../jpeglib.h"
#include "../../jsimd.h"
#include "../../jdct.h"
#include "../../jsimddct.h"
#include "../jsimd.h"

#include <arm_mve.h>


/* jsimd_fdct_ifast_helium() performs a fast, not so accurate forward DCT
 * (Discrete Cosine Transform) on one block of samples.  It uses the same
 * calculations and produces exactly the same output as IJG's original
 * jpeg_fdct_ifast() function, which can be found in jfdctfst.c.
 *
 * Scaled integer constants are used to avoid floating-point arithmetic:
 *    0.382683433 = 12544 * 2^-15
 *    0.541196100 = 17795 * 2^-15
 *    0.707106781 = 23168 * 2^-15
 *    0.306562965 =  9984 * 2^-15
 *
 * See jfdctfst.c for further details of the DCT algorithm.  Where possible,
 * the variable names and comments here in jsimd_fdct_ifast_helium() match up
 * with those in jpeg_fdct_ifast().
 */

#define F_0_382  12544
#define F_0_541  17792
#define F_0_707  23168
#define F_0_306  9984


void jsimd_fdct_ifast_helium(DCTELEM *data)
{
  /* Load an 8x8 block of samples into Neon registers.  Gather loads
   * are used such that we have a column of samples per vector - allowing
   * all rows to be processed at once.
   */
  const uint16x8_t offsets = vidupq_n_u16(0, DCTSIZE);

  int16x8_t col0 = vldrhq_gather_shifted_offset_s16(data + 0, offsets);
  int16x8_t col1 = vldrhq_gather_shifted_offset_s16(data + 1, offsets);
  int16x8_t col2 = vldrhq_gather_shifted_offset_s16(data + 2, offsets);
  int16x8_t col3 = vldrhq_gather_shifted_offset_s16(data + 3, offsets);
  int16x8_t col4 = vldrhq_gather_shifted_offset_s16(data + 4, offsets);
  int16x8_t col5 = vldrhq_gather_shifted_offset_s16(data + 5, offsets);
  int16x8_t col6 = vldrhq_gather_shifted_offset_s16(data + 6, offsets);
  int16x8_t col7 = vldrhq_gather_shifted_offset_s16(data + 7, offsets);

  /* Pass 1: process rows. */

  int16x8_t tmp0 = vaddq_s16(col0, col7);
  int16x8_t tmp7 = vsubq_s16(col0, col7);
  int16x8_t tmp1 = vaddq_s16(col1, col6);
  int16x8_t tmp6 = vsubq_s16(col1, col6);
  int16x8_t tmp2 = vaddq_s16(col2, col5);
  int16x8_t tmp5 = vsubq_s16(col2, col5);
  int16x8_t tmp3 = vaddq_s16(col3, col4);
  int16x8_t tmp4 = vsubq_s16(col3, col4);

  /* Even part */
  int16x8_t tmp10 = vaddq_s16(tmp0, tmp3);    /* phase 2 */
  int16x8_t tmp13 = vsubq_s16(tmp0, tmp3);
  int16x8_t tmp11 = vaddq_s16(tmp1, tmp2);
  int16x8_t tmp12 = vsubq_s16(tmp1, tmp2);

  col0 = vaddq_s16(tmp10, tmp11);             /* phase 3 */
  col4 = vsubq_s16(tmp10, tmp11);

  int16x8_t z1 = vqdmulhq_n_s16(vaddq_s16(tmp12, tmp13), F_0_707);
  col2 = vaddq_s16(tmp13, z1);                /* phase 5 */
  col6 = vsubq_s16(tmp13, z1);

  /* Odd part */
  tmp10 = vaddq_s16(tmp4, tmp5);              /* phase 2 */
  tmp11 = vaddq_s16(tmp5, tmp6);
  tmp12 = vaddq_s16(tmp6, tmp7);

  int16x8_t z5 = vqdmulhq_n_s16(vsubq_s16(tmp10, tmp12), F_0_382);
  int16x8_t z2 = vqdmlahq_n_s16(z5, tmp10, F_0_541);
  int16x8_t z4 = vqdmlahq_n_s16(tmp12, tmp12, F_0_306);
  z4 = vaddq_s16(z4, z5);
  int16x8_t z3 = vqdmulhq_n_s16(tmp11, F_0_707);

  int16x8_t z11 = vaddq_s16(tmp7, z3);        /* phase 5 */
  int16x8_t z13 = vsubq_s16(tmp7, z3);

  col5 = vaddq_s16(z13, z2);                  /* phase 6 */
  col3 = vsubq_s16(z13, z2);
  col1 = vaddq_s16(z11, z4);
  col7 = vsubq_s16(z11, z4);

  /* Transpose to work on columns in pass 2. */
  vstrhq_scatter_shifted_offset_s16(data + 0, offsets, col0);
  vstrhq_scatter_shifted_offset_s16(data + 1, offsets, col1);
  vstrhq_scatter_shifted_offset_s16(data + 2, offsets, col2);
  vstrhq_scatter_shifted_offset_s16(data + 3, offsets, col3);
  vstrhq_scatter_shifted_offset_s16(data + 4, offsets, col4);
  vstrhq_scatter_shifted_offset_s16(data + 5, offsets, col5);
  vstrhq_scatter_shifted_offset_s16(data + 6, offsets, col6);
  vstrhq_scatter_shifted_offset_s16(data + 7, offsets, col7);

  int16x8_t row0 = vld1q_s16(data + 0 * DCTSIZE);
  int16x8_t row1 = vld1q_s16(data + 1 * DCTSIZE);
  int16x8_t row2 = vld1q_s16(data + 2 * DCTSIZE);
  int16x8_t row3 = vld1q_s16(data + 3 * DCTSIZE);
  int16x8_t row4 = vld1q_s16(data + 4 * DCTSIZE);
  int16x8_t row5 = vld1q_s16(data + 5 * DCTSIZE);
  int16x8_t row6 = vld1q_s16(data + 6 * DCTSIZE);
  int16x8_t row7 = vld1q_s16(data + 7 * DCTSIZE);

  /* Pass 2: process columns. */

  tmp0 = vaddq_s16(row0, row7);
  tmp7 = vsubq_s16(row0, row7);
  tmp1 = vaddq_s16(row1, row6);
  tmp6 = vsubq_s16(row1, row6);
  tmp2 = vaddq_s16(row2, row5);
  tmp5 = vsubq_s16(row2, row5);
  tmp3 = vaddq_s16(row3, row4);
  tmp4 = vsubq_s16(row3, row4);

  /* Even part */
  tmp10 = vaddq_s16(tmp0, tmp3);              /* phase 2 */
  tmp13 = vsubq_s16(tmp0, tmp3);
  tmp11 = vaddq_s16(tmp1, tmp2);
  tmp12 = vsubq_s16(tmp1, tmp2);

  row0 = vaddq_s16(tmp10, tmp11);             /* phase 3 */
  row4 = vsubq_s16(tmp10, tmp11);

  z1 = vqdmulhq_n_s16(vaddq_s16(tmp12, tmp13), F_0_707);
  row2 = vaddq_s16(tmp13, z1);                /* phase 5 */
  row6 = vsubq_s16(tmp13, z1);

  /* Odd part */
  tmp10 = vaddq_s16(tmp4, tmp5);              /* phase 2 */
  tmp11 = vaddq_s16(tmp5, tmp6);
  tmp12 = vaddq_s16(tmp6, tmp7);

  z5 = vqdmulhq_n_s16(vsubq_s16(tmp10, tmp12), F_0_382);
  z2 = vqdmlahq_n_s16(z5, tmp10, F_0_541);
  z4 = vqdmlahq_n_s16(tmp12, tmp12, F_0_306);
  z4 = vaddq_s16(z4, z5);
  z3 = vqdmulhq_n_s16(tmp11, F_0_707);

  z11 = vaddq_s16(tmp7, z3);                  /* phase 5 */
  z13 = vsubq_s16(tmp7, z3);

  row5 = vaddq_s16(z13, z2);                  /* phase 6 */
  row3 = vsubq_s16(z13, z2);
  row1 = vaddq_s16(z11, z4);
  row7 = vsubq_s16(z11, z4);

  vst1q_s16(data + 0 * DCTSIZE, row0);
  vst1q_s16(data + 1 * DCTSIZE, row1);
  vst1q_s16(data + 2 * DCTSIZE, row2);
  vst1q_s16(data + 3 * DCTSIZE, row3);
  vst1q_s16(data + 4 * DCTSIZE, row4);
  vst1q_s16(data + 5 * DCTSIZE, row5);
  vst1q_s16(data + 6 * DCTSIZE, row6);
  vst1q_s16(data + 7 * DCTSIZE, row7);
}
