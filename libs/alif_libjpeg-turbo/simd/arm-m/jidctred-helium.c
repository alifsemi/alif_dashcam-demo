/*
 * jidctred-helium.c - reduced-size IDCT (Arm Helium)
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


#define CONST_BITS  13
#define PASS1_BITS  2

#define F_0_211  1730
#define F_0_509  4176
#define F_0_601  4926
#define F_0_720  5906
#define F_0_765  6270
#define F_0_850  6967
#define F_0_899  7373
#define F_1_061  8697
#define F_1_272  10426
#define F_1_451  11893
#define F_1_847  15137
#define F_2_172  17799
#define F_2_562  20995
#define F_3_624  29692


/* jsimd_idct_2x2_helium() is an inverse DCT function that produces reduced-size
 * 2x2 output from an 8x8 DCT block.  It uses the same calculations and
 * produces exactly the same output as IJG's original jpeg_idct_2x2() function
 * from jpeg-6b, which can be found in jidctred.c.
 *
 * Scaled integer constants are used to avoid floating-point arithmetic:
 *    0.720959822 =  5906 * 2^-13
 *    0.850430095 =  6967 * 2^-13
 *    1.272758580 = 10426 * 2^-13
 *    3.624509785 = 29692 * 2^-13
 *
 * See jidctred.c for further details of the 2x2 IDCT algorithm.  Where
 * possible, the variable names and comments here in jsimd_idct_2x2_helium()
 * match up with those in jpeg_idct_2x2().
 */

void jsimd_idct_2x2_helium(void * restrict dct_table, JCOEFPTR restrict coef_block,
                           JSAMPARRAY restrict output_buf, JDIMENSION output_col)
{
  ISLOW_MULT_TYPE *quantptr = dct_table;

  /* Load DCT coefficients. */
  int16x8_t row0 = vld1q_s16(coef_block + 0 * DCTSIZE);
  int16x8_t row1 = vld1q_s16(coef_block + 1 * DCTSIZE);
  int16x8_t row3 = vld1q_s16(coef_block + 3 * DCTSIZE);
  int16x8_t row5 = vld1q_s16(coef_block + 5 * DCTSIZE);
  int16x8_t row7 = vld1q_s16(coef_block + 7 * DCTSIZE);

  /* Load quantization table values. */
  int16x8_t quant_row0 = vld1q_s16(quantptr + 0 * DCTSIZE);
  int16x8_t quant_row1 = vld1q_s16(quantptr + 1 * DCTSIZE);
  int16x8_t quant_row3 = vld1q_s16(quantptr + 3 * DCTSIZE);
  int16x8_t quant_row5 = vld1q_s16(quantptr + 5 * DCTSIZE);
  int16x8_t quant_row7 = vld1q_s16(quantptr + 7 * DCTSIZE);

  /* Dequantize DCT coefficients. */
  row0 = vmulq_s16(row0, quant_row0);
  row1 = vmulq_s16(row1, quant_row1);
  row3 = vmulq_s16(row3, quant_row3);
  row5 = vmulq_s16(row5, quant_row5);
  row7 = vmulq_s16(row7, quant_row7);

  /* Pass 1: process columns from input, put results in vectors row0 and
   * row1.
   */

  /* Even part */
  int32x4_t tmp10_b = vshllbq_n_s16(row0, CONST_BITS + 2);
  int32x4_t tmp10_t = vshlltq_n_s16(row0, CONST_BITS + 2);

  /* Odd part */
  int32x4_t tmp0_b = vmullbq_int_s16(row1, vdupq_n_s16(F_3_624));
  tmp0_b = vmlaq_n_s32(tmp0_b, vmovlbq_s16(row3), -F_1_272);
  tmp0_b = vmlaq_n_s32(tmp0_b, vmovlbq_s16(row5), F_0_850);
  tmp0_b = vmlaq_n_s32(tmp0_b, vmovlbq_s16(row7), -F_0_720);
  int32x4_t tmp0_t = vmulltq_int_s16(row1, vdupq_n_s16(F_3_624));
  tmp0_t = vmlaq_n_s32(tmp0_t, vmovltq_s16(row3), -F_1_272);
  tmp0_t = vmlaq_n_s32(tmp0_t, vmovltq_s16(row5), F_0_850);
  tmp0_t = vmlaq_n_s32(tmp0_t, vmovltq_s16(row7), -F_0_720);

  /* Final output stage: descale and narrow to 16-bit. */
  row0 = vrshrnq_n_s32(vaddq_s32(tmp10_b, tmp0_b),
                       vaddq_s32(tmp10_t, tmp0_t), CONST_BITS);
  row1 = vrshrnq_n_s32(vsubq_s32(tmp10_b, tmp0_b),
                       vsubq_s32(tmp10_t, tmp0_t), CONST_BITS);

  /* Shuffle everything around. We're going to use the bottom
   * two lanes of 32-bit vectors to hold the two rows, and only care
   * about columns 0,1,3,5,7.
   *
   * First manually place arrange column 0 as 16-bit lanes 0 and 2,
   * ready for a VSHLLB.S16.
   */
  int16x8_t col_0_0 = vsetq_lane_s16(vgetq_lane_s16(row1, 0), row0, 2);

  /* Then group 1357 pairs using interleaved store, and we use offset
   * loads to get a pair into the bottom lanes. Load/store is relatively
   * cheap on M55, as it gets interleaved with arithmetic.
   */

  int32x4x2_t cols_1357_1357 = { {
    vmovltq_s16(row0),
    vmovltq_s16(row1)
  } } ;

  int32_t cols_11335577[8 + 2]; /* 8 values, and we read off the end to get the last 2 */
  vst2q_s32(cols_11335577, cols_1357_1357);

  /* Pass 2: process two rows, store to output array. */

  /* Even part */

  int32x4_t tmp10 = vshllbq_n_s16(col_0_0, CONST_BITS + 2);

  /* Odd part */

  int32x4_t tmp0 = vmulq_n_s32(vld1q_s32(cols_11335577 + 0), F_3_624);
  tmp0 = vmlaq_n_s32(tmp0, vld1q_s32(cols_11335577 + 2), -F_1_272);
  tmp0 = vmlaq_n_s32(tmp0, vld1q_s32(cols_11335577 + 4), F_0_850);
  tmp0 = vmlaq_n_s32(tmp0, vld1q_s32(cols_11335577 + 6), -F_0_720);

  /* Final output stage. */
  int32x4_t out0 = vaddq_s32(tmp10, tmp0);
  int32x4_t out1 = vsubq_s32(tmp10, tmp0);

  /* Pack to 16 bits, dropping the low 16 bits. */
  int16x8_t output_s16 = vshrnbq_n_s32(vreinterpretq_s16_s32(out1), out0, 16);

  /* Complete the descale, clamping to [-128-127]. */
  uint8x16_t output_u8 = vreinterpretq_u8_s8(
                           vqrshrnbq_n_s16(vuninitializedq_s8(),
                                           output_s16,
                                           CONST_BITS + PASS1_BITS + 3 + 2 - 16));
  /* Recenter. */
  output_u8 = vaddq_n_u8(output_u8, CENTERJSAMPLE);

  /* Store 2x2 block to memory. */
  output_buf[0][output_col + 0] = vgetq_lane_u8(output_u8, 0);
  output_buf[0][output_col + 1] = vgetq_lane_u8(output_u8, 2);
  output_buf[1][output_col + 0] = vgetq_lane_u8(output_u8, 4);
  output_buf[1][output_col + 1] = vgetq_lane_u8(output_u8, 6);
}


/* jsimd_idct_4x4_helium() is an inverse DCT function that produces reduced-size
 * 4x4 output from an 8x8 DCT block.  It uses the same calculations and
 * produces exactly the same output as IJG's original jpeg_idct_4x4() function
 * from jpeg-6b, which can be found in jidctred.c.
 *
 * Scaled integer constants are used to avoid floating-point arithmetic:
 *    0.211164243 =  1730 * 2^-13
 *    0.509795579 =  4176 * 2^-13
 *    0.601344887 =  4926 * 2^-13
 *    0.765366865 =  6270 * 2^-13
 *    0.899976223 =  7373 * 2^-13
 *    1.061594337 =  8697 * 2^-13
 *    1.451774981 = 11893 * 2^-13
 *    1.847759065 = 15137 * 2^-13
 *    2.172734803 = 17799 * 2^-13
 *    2.562915447 = 20995 * 2^-13
 *
 * See jidctred.c for further details of the 4x4 IDCT algorithm.  Where
 * possible, the variable names and comments here in jsimd_idct_4x4_helium()
 * match up with those in jpeg_idct_4x4().
 */

void jsimd_idct_4x4_helium(void * restrict dct_table, JCOEFPTR restrict coef_block,
                           JSAMPARRAY restrict output_buf, JDIMENSION output_col)
{
#define quant_s32(x, y) vldrhq_s32(quantptr + (y) * DCTSIZE + (x))
#define quant_s16(y) vldrhq_s16(quantptr + (y) * DCTSIZE)
#define row_s16(y) vldrhq_s16(coef_block + (y) * DCTSIZE)
#define row_s32(x, y) vldrhq_s32(coef_block + (y) * DCTSIZE + (x))

  ISLOW_MULT_TYPE *quantptr = dct_table;

  /* Dequantize DC coefficients. */
  uint16x8_t row0 = vmulq_s16(row_s16(0), quant_s16(0));
  uint16x8_t row1, row2, row3;

  /* Construct bitmap to test if all AC coefficients are 0. */
  int16x8_t bitmap = vorrq_s16(row_s16(7), row_s16(6));
  bitmap = vorrq_s16(bitmap, row_s16(5));
  bitmap = vorrq_s16(bitmap, row_s16(3)); // we ignore row 4
  bitmap = vorrq_s16(bitmap, row_s16(2));
  bitmap = vorrq_s16(bitmap, row_s16(1));
  mve_pred16_t ac_bitmap = vcmpneq_n_s16(bitmap, 0);

  if (ac_bitmap == 0) {
    /* All AC coefficients are zero.
     * Compute DC values and duplicate into row vectors 0, 1, 2, and 3.
     */
    int16x8_t dcval = vshlq_n_s16(row0, PASS1_BITS);
    row0 = dcval;
    row1 = dcval;
    row2 = dcval;
    row3 = dcval;
  } else if ((ac_bitmap & 0xCCCC) == 0) {
    /* AC coefficients are zero for columns 1, 3, 5 and 7.
     * Compute DC values for these columns.
     * (Try this first, because it's moderately likely, and
     * it's easier to process 0+2+4+6 than 0+1+2+3 with Helium)
     */
    int16x8_t dcval = vshlq_n_s16(row0, PASS1_BITS);

    /* Commence regular IDCT computation for columns 0, 2, 4, and 6. */

    /* Even part */
    int32x4_t tmp0 = vshllbq_n_s16(row0, CONST_BITS + 1);

    int32x4_t z2 = vmullbq_int_s16(row_s16(2), quant_s16(2));
    int32x4_t z3 = vmullbq_int_s16(row_s16(6), quant_s16(6));

    int32x4_t tmp2 = vmulq_n_s32(z2, F_1_847);
    tmp2 = vmlaq_n_s32(tmp2, z3, -F_0_765);

    int32x4_t tmp10 = vaddq_s32(tmp0, tmp2);
    int32x4_t tmp12 = vsubq_s32(tmp0, tmp2);

    /* Odd part */
    int32x4_t z1 = vmullbq_int_s16(row_s16(7), quant_s16(7));
    z2 = vmullbq_int_s16(row_s16(5), quant_s16(5));
    z3 = vmullbq_int_s16(row_s16(3), quant_s16(3));
    int32x4_t z4 = vmullbq_int_s16(row_s16(1), quant_s16(1));

    tmp0 = vmulq_n_s32(z1, -F_0_211);
    tmp0 = vmlaq_n_s32(tmp0, z2, F_1_451);
    tmp0 = vmlaq_n_s32(tmp0, z3, -F_2_172);
    tmp0 = vmlaq_n_s32(tmp0, z4, F_1_061);

    tmp2 = vmulq_n_s32(z1, -F_0_509);
    tmp2 = vmlaq_n_s32(tmp2, z2, -F_0_601);
    tmp2 = vmlaq_n_s32(tmp2, z3, F_0_899);
    tmp2 = vmlaq_n_s32(tmp2, z4, F_2_562);

    /* Final output stage: descale and narrow to 16-bit. */
    row0 = vrshrnbq_n_s32(dcval,
                          vaddq_s32(tmp10, tmp2),
                          CONST_BITS - PASS1_BITS + 1);
    row3 = vrshrnbq_n_s32(dcval,
                          vsubq_s32(tmp10, tmp2),
                          CONST_BITS - PASS1_BITS + 1);
    row1 = vrshrnbq_n_s32(dcval,
                          vaddq_s32(tmp12, tmp0),
                          CONST_BITS - PASS1_BITS + 1);
    row2 = vrshrnbq_n_s32(dcval,
                          vsubq_s32(tmp12, tmp0),
                          CONST_BITS - PASS1_BITS + 1);
  } else if ((ac_bitmap & 0x3333) == 0) {
    /* AC coefficients are zero for columns 0, 2, 4 and 6.
     * Compute DC values for these columns.
     * (Not terribly likely, but as we're taking the
     * time to check the bitmap anyway)
     */
    int16x8_t dcval = vshlq_n_s16(row0, PASS1_BITS);

    /* Commence regular IDCT computation for columns 1, 3, 5, and 7. */

    /* Even part */
    int32x4_t tmp0 = vshlltq_n_s16(row0, CONST_BITS + 1);

    int32x4_t z2 = vmulltq_int_s16(row_s16(2), quant_s16(2));
    int32x4_t z3 = vmulltq_int_s16(row_s16(6), quant_s16(6));

    int32x4_t tmp2 = vmulq_n_s32(z2, F_1_847);
    tmp2 = vmlaq_n_s32(tmp2, z3, -F_0_765);

    int32x4_t tmp10 = vaddq_s32(tmp0, tmp2);
    int32x4_t tmp12 = vsubq_s32(tmp0, tmp2);

    /* Odd part */
    int32x4_t z1 = vmulltq_int_s16(row_s16(7), quant_s16(7));
    z2 = vmulltq_int_s16(row_s16(5), quant_s16(5));
    z3 = vmulltq_int_s16(row_s16(3), quant_s16(3));
    int32x4_t z4 = vmulltq_int_s16(row_s16(1), quant_s16(1));

    tmp0 = vmulq_n_s32(z1, -F_0_211);
    tmp0 = vmlaq_n_s32(tmp0, z2, F_1_451);
    tmp0 = vmlaq_n_s32(tmp0, z3, -F_2_172);
    tmp0 = vmlaq_n_s32(tmp0, z4, F_1_061);

    tmp2 = vmulq_n_s32(z1, -F_0_509);
    tmp2 = vmlaq_n_s32(tmp2, z2, -F_0_601);
    tmp2 = vmlaq_n_s32(tmp2, z3, F_0_899);
    tmp2 = vmlaq_n_s32(tmp2, z4, F_2_562);

    /* Final output stage: descale and narrow to 16-bit. */
    row0 = vrshrntq_n_s32(dcval,
                          vaddq_s32(tmp10, tmp2),
                          CONST_BITS - PASS1_BITS + 1);
    row3 = vrshrntq_n_s32(dcval,
                          vsubq_s32(tmp10, tmp2),
                          CONST_BITS - PASS1_BITS + 1);
    row1 = vrshrntq_n_s32(dcval,
                          vaddq_s32(tmp12, tmp0),
                          CONST_BITS - PASS1_BITS + 1);
    row2 = vrshrntq_n_s32(dcval,
                          vsubq_s32(tmp12, tmp0),
                          CONST_BITS - PASS1_BITS + 1);
  } else if ((ac_bitmap & 0xFF00) == 0 && 0) {
    /* AC coefficients are zero for columns 4, 5, 6, and 7.
     * Compute DC values for these columns.
     * (This is very likely, but it's a pain to arrange the data.
     * Interestingly, I thought this memory pack/unpack was a
     * reasonable way to do it, but Arm Compiler 6.19 chooses to
     * transform it into a heap of register moves. And it actually
     * ends up slower, due to the sheer amount of shuffling.
     */
    int16_t store[8];
    int16x8_t dcval = vshlq_n_s16(row0, PASS1_BITS);
    vst1q_s16(store, row0);

    /* Commence regular IDCT computation for columns 0, 1, 2, and 3. */

    /* Even part */
    int32x4_t tmp0 = vshlq_n_s32(vldrhq_s32(store), CONST_BITS - PASS1_BITS + 1 );

    int32x4_t z2 = vmulq_s32(row_s32(0, 2), quant_s32(0, 2));
    int32x4_t z3 = vmulq_s32(row_s32(0, 6), quant_s32(0, 6));

    int32x4_t tmp2 = vmulq_n_s32(z2, F_1_847);
    tmp2 = vmlaq_n_s32(tmp2, z3, -F_0_765);

    int32x4_t tmp10 = vaddq_s32(tmp0, tmp2);
    int32x4_t tmp12 = vsubq_s32(tmp0, tmp2);

    /* Odd part */
    int32x4_t z1 = vmulq_s32(row_s32(0, 7), quant_s32(0, 7));
    z2 = vmulq_s32(row_s32(0, 5), quant_s32(0, 5));
    z3 = vmulq_s32(row_s32(0, 3), quant_s32(0, 3));
    int32x4_t z4 = vmulq_s32(row_s32(0, 1), quant_s32(0, 1));

    tmp0 = vmulq_n_s32(z1, -F_0_211);
    tmp0 = vmlaq_n_s32(tmp0, z2, F_1_451);
    tmp0 = vmlaq_n_s32(tmp0, z3, -F_2_172);
    tmp0 = vmlaq_n_s32(tmp0, z4, F_1_061);

    tmp2 = vmulq_n_s32(z1, -F_0_509);
    tmp2 = vmlaq_n_s32(tmp2, z2, -F_0_601);
    tmp2 = vmlaq_n_s32(tmp2, z3, F_0_899);
    tmp2 = vmlaq_n_s32(tmp2, z4, F_2_562);

    /* Final output stage: descale and narrow to 16-bit. */
    /* Store into 4 entries of temp buffer, reloading 8 to get dc values */
    vstrhq_s32(store, vrshrq_n_s32(vaddq_s32(tmp10, tmp2),
                                   CONST_BITS - PASS1_BITS + 1));
    row0 = vldrhq_s16(store);
    vstrhq_s32(store, vrshrq_n_s32(vsubq_s32(tmp10, tmp2),
                                   CONST_BITS - PASS1_BITS + 1));
    row3 = vldrhq_s16(store);
    vstrhq_s32(store, vrshrq_n_s32(vaddq_s32(tmp12, tmp0),
                                   CONST_BITS - PASS1_BITS + 1));
    row1 = vldrhq_s16(store);
    vstrhq_s32(store, vrshrq_n_s32(vsubq_s32(tmp12, tmp0),
                                   CONST_BITS - PASS1_BITS + 1));
    row2 = vldrhq_s16(store);
  } else {
    /* Many AC coefficients are non-zero; full IDCT calculation required. */

    /* Even part */
    int32x4_t tmp0_b = vshllbq_n_s16(row0, CONST_BITS + 1);
    int32x4_t tmp0_t = vshlltq_n_s16(row0, CONST_BITS + 1);

    int16x8_t z2 = vmulq_s16(row_s16(2), quant_s16(2));
    int16x8_t z3 = vmulq_s16(row_s16(6), quant_s16(6));

    int32x4_t tmp2_b = vmullbq_int_s16(z2, vdupq_n_s16(F_1_847));
    int32x4_t tmp2_t = vmulltq_int_s16(z2, vdupq_n_s16(F_1_847));
    tmp2_b = vmlaq_n_s32(tmp2_b, vmovlbq_s16(z3), -F_0_765);
    tmp2_t = vmlaq_n_s32(tmp2_t, vmovltq_s16(z3), -F_0_765);

    int32x4_t tmp10_b = vaddq_s32(tmp0_b, tmp2_b);
    int32x4_t tmp10_t = vaddq_s32(tmp0_t, tmp2_t);
    int32x4_t tmp12_b = vsubq_s32(tmp0_b, tmp2_b);
    int32x4_t tmp12_t = vsubq_s32(tmp0_t, tmp2_t);

    /* Odd part */
    int16x8_t z1 = vmulq_s16(row_s16(7), quant_s16(7));
    z2 = vmulq_s16(row_s16(5), quant_s16(5));
    z3 = vmulq_s16(row_s16(3), quant_s16(3));
    int16x8_t z4 = vmulq_s16(row_s16(1), quant_s16(1));

    tmp0_b = vmullbq_int_s16(z1, vdupq_n_s16(-F_0_211));
    tmp0_b = vmlaq_n_s32(tmp0_b, vmovlbq_s16(z2), F_1_451);
    tmp0_b = vmlaq_n_s32(tmp0_b, vmovlbq_s16(z3), -F_2_172);
    tmp0_b = vmlaq_n_s32(tmp0_b, vmovlbq_s16(z4), F_1_061);
    tmp0_t = vmulltq_int_s16(z1, vdupq_n_s16(-F_0_211));
    tmp0_t = vmlaq_n_s32(tmp0_t, vmovltq_s16(z2), F_1_451);
    tmp0_t = vmlaq_n_s32(tmp0_t, vmovltq_s16(z3), -F_2_172);
    tmp0_t = vmlaq_n_s32(tmp0_t, vmovltq_s16(z4), F_1_061);

    tmp2_b = vmullbq_int_s16(z1, vdupq_n_s16(-F_0_509));
    tmp2_b = vmlaq_n_s32(tmp2_b, vmovlbq_s16(z2), -F_0_601);
    tmp2_b = vmlaq_n_s32(tmp2_b, vmovlbq_s16(z3), F_0_899);
    tmp2_b = vmlaq_n_s32(tmp2_b, vmovlbq_s16(z4), F_2_562);
    tmp2_t = vmulltq_int_s16(z1, vdupq_n_s16(-F_0_509));
    tmp2_t = vmlaq_n_s32(tmp2_t, vmovltq_s16(z2), -F_0_601);
    tmp2_t = vmlaq_n_s32(tmp2_t, vmovltq_s16(z3), F_0_899);
    tmp2_t = vmlaq_n_s32(tmp2_t, vmovltq_s16(z4), F_2_562);

    /* Final output stage: descale and narrow to 16-bit. */
    row0 = vrshrnq_n_s32(vaddq_s32(tmp10_b, tmp2_b),
                         vaddq_s32(tmp10_t, tmp2_t), CONST_BITS - PASS1_BITS + 1);
    row3 = vrshrnq_n_s32(vsubq_s32(tmp10_b, tmp2_b),
                         vsubq_s32(tmp10_t, tmp2_t), CONST_BITS - PASS1_BITS + 1);
    row1 = vrshrnq_n_s32(vaddq_s32(tmp12_b, tmp0_b),
                         vaddq_s32(tmp12_t, tmp0_t), CONST_BITS - PASS1_BITS + 1);
    row2 = vrshrnq_n_s32(vsubq_s32(tmp12_b, tmp0_b),
                         vsubq_s32(tmp12_t, tmp0_t), CONST_BITS - PASS1_BITS + 1);
  }

  /* Transpose 8x4 block to perform IDCT on rows in second pass. */
  int16_t cols[4 * 8];
  int16x8x4_t rows = { {
    row0, row1, row2, row3
  } };
  vst4q_s16(cols, rows);

#define col_s32(x) vldrhq_s32(cols + 4 * (x))

  /* Commence second pass of IDCT. */

  /* Even part */
  int32x4_t tmp0 = vshlq_n_s32(col_s32(0), CONST_BITS + 1);
  int32x4_t tmp2 = vmulq_n_s32(col_s32(2), F_1_847);
  tmp2 = vmlaq_n_s32(tmp2, col_s32(6), -F_0_765);

  int32x4_t tmp10 = vaddq_s32(tmp0, tmp2);
  int32x4_t tmp12 = vsubq_s32(tmp0, tmp2);

  /* Odd part */
  tmp0 = vmulq_n_s32(col_s32(7), -F_0_211);
  tmp0 = vmlaq_n_s32(tmp0, col_s32(5), F_1_451);
  tmp0 = vmlaq_n_s32(tmp0, col_s32(3), -F_2_172);
  tmp0 = vmlaq_n_s32(tmp0, col_s32(1), F_1_061);

  tmp2 = vmulq_n_s32(col_s32(7), -F_0_509);
  tmp2 = vmlaq_n_s32(tmp2, col_s32(5), -F_0_601);
  tmp2 = vmlaq_n_s32(tmp2, col_s32(3), F_0_899);
  tmp2 = vmlaq_n_s32(tmp2, col_s32(1), F_2_562);

  /* Final output stage: pack to 16 bits, dropping the low 16 bits.*/
  int16x8_t output_cols_02 = vshrnbq_n_s32(vreinterpretq_s16_s32(
                               vsubq_s32(tmp12, tmp0)),
                               vaddq_s32(tmp10, tmp2), 16);
  int16x8_t output_cols_13 = vshrnbq_n_s32(vreinterpretq_s16_s32(
                               vsubq_s32(tmp10, tmp2)),
                               vaddq_s32(tmp12, tmp0), 16);
  /* Complete the descale, clamping to [-128-127]. */
  uint8x16_t output_cols_0123 = vreinterpretq_u8_s8(
                                  vqrshrnq_n_s16(
                                    output_cols_02, output_cols_13,
                                    CONST_BITS + PASS1_BITS + 3 + 1 - 16));
  /* Recenter. */
  output_cols_0123 = vaddq_n_u8(output_cols_0123, CENTERJSAMPLE);

  /* Store 4x4 block to memory. */
  uint32x4_t outptr_0123 = vld1q_u32((uint32_t *) output_buf);

  /* We want to store at vector outptr_0123 + scalar output_col.
   * vstrwq_scatter_base_u32 requires constant offset for VSTRW.U32 Qd,[Qn,#imm],
   * so can't be used.
   * vstrwq_scatter_offset_u32 assumes that the scalar is the base in VSTRW.U32 Qd,[Rn,Qm],
   * so we have to cast our offset to look like a pointer. The actual instruction
   * is agnostic as to the contents of the vector and scalar - it just adds.
   */
  vstrwq_scatter_offset_u32((uint32_t *) output_col, outptr_0123,
                            vreinterpretq_u32_u8(output_cols_0123));

}
