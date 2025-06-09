/*
 * jfdctint-helium.c - accurate integer FDCT (Arm Helium)
 *
 * Copyright (C) 2020, Arm Limited.  All Rights Reserved.
 * Copyright (C) 2020, D. R. Commander.  All Rights Reserved.
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
#include "helium-helpers.h"

#include <arm_mve.h>


/* jsimd_fdct_islow_helium() performs a slower but more accurate forward DCT
 * (Discrete Cosine Transform) on one block of samples.  It uses the same
 * calculations and produces exactly the same output as IJG's original
 * jpeg_fdct_islow() function, which can be found in jfdctint.c.
 *
 * Scaled integer constants are used to avoid floating-point arithmetic:
 *    0.298631336 =  2446 * 2^-13
 *    0.390180644 =  3196 * 2^-13
 *    0.541196100 =  4433 * 2^-13
 *    0.765366865 =  6270 * 2^-13
 *    0.899976223 =  7373 * 2^-13
 *    1.175875602 =  9633 * 2^-13
 *    1.501321110 = 12299 * 2^-13
 *    1.847759065 = 15137 * 2^-13
 *    1.961570560 = 16069 * 2^-13
 *    2.053119869 = 16819 * 2^-13
 *    2.562915447 = 20995 * 2^-13
 *    3.072711026 = 25172 * 2^-13
 *
 * See jfdctint.c for further details of the DCT algorithm.  Where possible,
 * the variable names and comments here in jsimd_fdct_islow_helium() match up
 * with those in jpeg_fdct_islow().
 */

#define CONST_BITS  13
#define PASS1_BITS  2

#define DESCALE_P1  (CONST_BITS - PASS1_BITS)
#define DESCALE_P2  (CONST_BITS + PASS1_BITS)

#define F_0_298  2446
#define F_0_390  3196
#define F_0_541  4433
#define F_0_765  6270
#define F_0_899  7373
#define F_1_175  9633
#define F_1_501  12299
#define F_1_847  15137
#define F_1_961  16069
#define F_2_053  16819
#define F_2_562  20995
#define F_3_072  25172


void jsimd_fdct_islow_helium(DCTELEM *data)
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
  int16x8_t tmp10 = vaddq_s16(tmp0, tmp3);
  int16x8_t tmp13 = vsubq_s16(tmp0, tmp3);
  int16x8_t tmp11 = vaddq_s16(tmp1, tmp2);
  int16x8_t tmp12 = vsubq_s16(tmp1, tmp2);

  col0 = vshlq_n_s16(vaddq_s16(tmp10, tmp11), PASS1_BITS);
  col4 = vshlq_n_s16(vsubq_s16(tmp10, tmp11), PASS1_BITS);

  int16x8_t tmp12_add_tmp13 = vaddq_s16(tmp12, tmp13);
  int32x4_t z1_b =
    vmullbq_int_s16(tmp12_add_tmp13, vdupq_n_s16(F_0_541));
  int32x4_t z1_t =
    vmulltq_int_s16(tmp12_add_tmp13, vdupq_n_s16(F_0_541));

  int32x4_t col2_scaled_b =
    vmlaq_n_s32(z1_b, vmovlbq_s16(tmp13), F_0_765);
  int32x4_t col2_scaled_t =
    vmlaq_n_s32(z1_t, vmovltq_s16(tmp13), F_0_765);
  col2 = vrshrnq_n_s32(col2_scaled_b, col2_scaled_t, DESCALE_P1);

  int32x4_t col6_scaled_b =
    vmlaq_n_s32(z1_b, vmovlbq_s16(tmp12), -F_1_847);
  int32x4_t col6_scaled_t =
    vmlaq_n_s32(z1_t, vmovltq_s16(tmp12), -F_1_847);
  col6 = vrshrnq_n_s32(col6_scaled_b, col6_scaled_t, DESCALE_P1);

  /* Odd part */
  int16x8_t z1 = vaddq_s16(tmp4, tmp7);
  int16x8_t z2 = vaddq_s16(tmp5, tmp6);
  int16x8_t z3 = vaddq_s16(tmp4, tmp6);
  int16x8_t z4 = vaddq_s16(tmp5, tmp7);
  /* sqrt(2) * c3 */
  int32x4_t z5_b = vmullbq_int_s16(z3, vdupq_n_s16(F_1_175));
  int32x4_t z5_t = vmulltq_int_s16(z3, vdupq_n_s16(F_1_175));
  z5_b = vmlaq_n_s32(z5_b, vmovlbq_s16(z4), F_1_175);
  z5_t = vmlaq_n_s32(z5_t, vmovltq_s16(z4), F_1_175);

  /* sqrt(2) * (-c1+c3+c5-c7) */
  int32x4_t tmp4_b = vmullbq_int_s16(tmp4, vdupq_n_s16(F_0_298));
  int32x4_t tmp4_t = vmulltq_int_s16(tmp4, vdupq_n_s16(F_0_298));
  /* sqrt(2) * ( c1+c3-c5+c7) */
  int32x4_t tmp5_b = vmullbq_int_s16(tmp5, vdupq_n_s16(F_2_053));
  int32x4_t tmp5_t = vmulltq_int_s16(tmp5, vdupq_n_s16(F_2_053));
  /* sqrt(2) * ( c1+c3+c5-c7) */
  int32x4_t tmp6_b = vmullbq_int_s16(tmp6, vdupq_n_s16(F_3_072));
  int32x4_t tmp6_t = vmulltq_int_s16(tmp6, vdupq_n_s16(F_3_072));
  /* sqrt(2) * ( c1+c3-c5-c7) */
  int32x4_t tmp7_b = vmullbq_int_s16(tmp7, vdupq_n_s16(F_1_501));
  int32x4_t tmp7_t = vmulltq_int_s16(tmp7, vdupq_n_s16(F_1_501));

  /* sqrt(2) * (c7-c3) */
  z1_b = vmullbq_int_s16(z1, vdupq_n_s16(-F_0_899));
  z1_t = vmulltq_int_s16(z1, vdupq_n_s16(-F_0_899));
  /* sqrt(2) * (-c1-c3) */
  int32x4_t z2_b = vmullbq_int_s16(z2, vdupq_n_s16(-F_2_562));
  int32x4_t z2_t = vmulltq_int_s16(z2, vdupq_n_s16(-F_2_562));
  /* sqrt(2) * (-c3-c5) */
  int32x4_t z3_b = vmullbq_int_s16(z3, vdupq_n_s16(-F_1_961));
  int32x4_t z3_t = vmulltq_int_s16(z3, vdupq_n_s16(-F_1_961));
  /* sqrt(2) * (c5-c3) */
  int32x4_t z4_b = vmullbq_int_s16(z4, vdupq_n_s16(-F_0_390));
  int32x4_t z4_t = vmulltq_int_s16(z4, vdupq_n_s16(-F_0_390));

  z3_b = vaddq_s32(z3_b, z5_b);
  z3_t = vaddq_s32(z3_t, z5_t);
  z4_b = vaddq_s32(z4_b, z5_b);
  z4_t = vaddq_s32(z4_t, z5_t);

  tmp4_b = vaddq_s32(tmp4_b, z1_b);
  tmp4_t = vaddq_s32(tmp4_t, z1_t);
  tmp4_b = vaddq_s32(tmp4_b, z3_b);
  tmp4_t = vaddq_s32(tmp4_t, z3_t);
  col7 = vrshrnq_n_s32(tmp4_b, tmp4_t, DESCALE_P1);

  tmp5_b = vaddq_s32(tmp5_b, z2_b);
  tmp5_t = vaddq_s32(tmp5_t, z2_t);
  tmp5_b = vaddq_s32(tmp5_b, z4_b);
  tmp5_t = vaddq_s32(tmp5_t, z4_t);
  col5 = vrshrnq_n_s32(tmp5_b, tmp5_t, DESCALE_P1);

  tmp6_b = vaddq_s32(tmp6_b, z2_b);
  tmp6_t = vaddq_s32(tmp6_t, z2_t);
  tmp6_b = vaddq_s32(tmp6_b, z3_b);
  tmp6_t = vaddq_s32(tmp6_t, z3_t);
  col3 = vrshrnq_n_s32(tmp6_b, tmp6_t, DESCALE_P1);

  tmp7_b = vaddq_s32(tmp7_b, z1_b);
  tmp7_t = vaddq_s32(tmp7_t, z1_t);
  tmp7_b = vaddq_s32(tmp7_b, z4_b);
  tmp7_t = vaddq_s32(tmp7_t, z4_t);
  col1 = vrshrnq_n_s32(tmp7_b, tmp7_t, DESCALE_P1);

  /* Store back columns for pass 2 on rows. */
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
  tmp10 = vaddq_s16(tmp0, tmp3);
  tmp13 = vsubq_s16(tmp0, tmp3);
  tmp11 = vaddq_s16(tmp1, tmp2);
  tmp12 = vsubq_s16(tmp1, tmp2);

  row0 = vrshrq_n_s16(vaddq_s16(tmp10, tmp11), PASS1_BITS);
  row4 = vrshrq_n_s16(vsubq_s16(tmp10, tmp11), PASS1_BITS);

  tmp12_add_tmp13 = vaddq_s16(tmp12, tmp13);
  z1_b = vmullbq_int_s16(tmp12_add_tmp13, vdupq_n_s16(F_0_541));
  z1_t = vmulltq_int_s16(tmp12_add_tmp13, vdupq_n_s16(F_0_541));

  int32x4_t row2_scaled_b =
    vmlaq_n_s32(z1_b, vmovlbq_s16(tmp13), F_0_765);
  int32x4_t row2_scaled_t =
    vmlaq_n_s32(z1_t, vmovltq_s16(tmp13), F_0_765);
  row2 = vrshrnq_n_s32(row2_scaled_b,  row2_scaled_t, DESCALE_P2);

  int32x4_t row6_scaled_b =
    vmlaq_n_s32(z1_b, vmovlbq_s16(tmp12), -F_1_847);
  int32x4_t row6_scaled_t =
    vmlaq_n_s32(z1_t, vmovltq_s16(tmp12), -F_1_847);
  row6 = vrshrnq_n_s32(row6_scaled_b, row6_scaled_t, DESCALE_P2);

  /* Odd part */
  z1 = vaddq_s16(tmp4, tmp7);
  z2 = vaddq_s16(tmp5, tmp6);
  z3 = vaddq_s16(tmp4, tmp6);
  z4 = vaddq_s16(tmp5, tmp7);
  /* sqrt(2) * c3 */
  z5_b = vmullbq_int_s16(z3, vdupq_n_s16(F_1_175));
  z5_t = vmulltq_int_s16(z3, vdupq_n_s16(F_1_175));
  z5_b = vmlaq_n_s32(z5_b, vmovlbq_s16(z4), F_1_175);
  z5_t = vmlaq_n_s32(z5_t, vmovltq_s16(z4), F_1_175);

  /* sqrt(2) * (-c1+c3+c5-c7) */
  tmp4_b = vmullbq_int_s16(tmp4, vdupq_n_s16(F_0_298));
  tmp4_t = vmulltq_int_s16(tmp4, vdupq_n_s16(F_0_298));
  /* sqrt(2) * ( c1+c3-c5+c7) */
  tmp5_b = vmullbq_int_s16(tmp5, vdupq_n_s16(F_2_053));
  tmp5_t = vmulltq_int_s16(tmp5, vdupq_n_s16(F_2_053));
  /* sqrt(2) * ( c1+c3+c5-c7) */
  tmp6_b = vmullbq_int_s16(tmp6, vdupq_n_s16(F_3_072));
  tmp6_t = vmulltq_int_s16(tmp6, vdupq_n_s16(F_3_072));
  /* sqrt(2) * ( c1+c3-c5-c7) */
  tmp7_b = vmullbq_int_s16(tmp7, vdupq_n_s16(F_1_501));
  tmp7_t = vmulltq_int_s16(tmp7, vdupq_n_s16(F_1_501));

  /* sqrt(2) * (c7-c3) */
  z1_b = vmullbq_int_s16(z1, vdupq_n_s16(-F_0_899));
  z1_t = vmulltq_int_s16(z1, vdupq_n_s16(-F_0_899));
  /* sqrt(2) * (-c1-c3) */
  z2_b = vmullbq_int_s16(z2, vdupq_n_s16(-F_2_562));
  z2_t = vmulltq_int_s16(z2, vdupq_n_s16(-F_2_562));
  /* sqrt(2) * (-c3-c5) */
  z3_b = vmullbq_int_s16(z3, vdupq_n_s16(-F_1_961));
  z3_t = vmulltq_int_s16(z3, vdupq_n_s16(-F_1_961));
  /* sqrt(2) * (c5-c3) */
  z4_b = vmullbq_int_s16(z4, vdupq_n_s16(-F_0_390));
  z4_t = vmulltq_int_s16(z4, vdupq_n_s16(-F_0_390));

  z3_b = vaddq_s32(z3_b, z5_b);
  z3_t = vaddq_s32(z3_t, z5_t);
  z4_b = vaddq_s32(z4_b, z5_b);
  z4_t = vaddq_s32(z4_t, z5_t);

  tmp4_b = vaddq_s32(tmp4_b, z1_b);
  tmp4_t = vaddq_s32(tmp4_t, z1_t);
  tmp4_b = vaddq_s32(tmp4_b, z3_b);
  tmp4_t = vaddq_s32(tmp4_t, z3_t);
  row7 = vrshrnq_n_s32(tmp4_b, tmp4_t, DESCALE_P2);

  tmp5_b = vaddq_s32(tmp5_b, z2_b);
  tmp5_t = vaddq_s32(tmp5_t, z2_t);
  tmp5_b = vaddq_s32(tmp5_b, z4_b);
  tmp5_t = vaddq_s32(tmp5_t, z4_t);
  row5 = vrshrnq_n_s32(tmp5_b, tmp5_t, DESCALE_P2);

  tmp6_b = vaddq_s32(tmp6_b, z2_b);
  tmp6_t = vaddq_s32(tmp6_t, z2_t);
  tmp6_b = vaddq_s32(tmp6_b, z3_b);
  tmp6_t = vaddq_s32(tmp6_t, z3_t);
  row3 = vrshrnq_n_s32(tmp6_b, tmp6_t, DESCALE_P2);

  tmp7_b = vaddq_s32(tmp7_b, z1_b);
  tmp7_t = vaddq_s32(tmp7_t, z1_t);
  tmp7_b = vaddq_s32(tmp7_b, z4_b);
  tmp7_t = vaddq_s32(tmp7_t, z4_t);
  row1 = vrshrnq_n_s32(tmp7_b, tmp7_t, DESCALE_P2);

  vst1q_s16(data + 0 * DCTSIZE, row0);
  vst1q_s16(data + 1 * DCTSIZE, row1);
  vst1q_s16(data + 2 * DCTSIZE, row2);
  vst1q_s16(data + 3 * DCTSIZE, row3);
  vst1q_s16(data + 4 * DCTSIZE, row4);
  vst1q_s16(data + 5 * DCTSIZE, row5);
  vst1q_s16(data + 6 * DCTSIZE, row6);
  vst1q_s16(data + 7 * DCTSIZE, row7);
}
