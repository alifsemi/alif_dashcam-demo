/*
 * jidctint-helium.c - accurate integer IDCT (Arm Helium)
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

#include <arm_mve.h>
#include "helium-helpers.h"


#define CONST_BITS  13
#define PASS1_BITS  2

#define DESCALE_P1  (CONST_BITS - PASS1_BITS)
#define DESCALE_P2  (CONST_BITS + PASS1_BITS + 3)

/* The computation of the inverse DCT requires the use of constants known at
 * compile time.  Scaled integer constants are used to avoid floating-point
 * arithmetic:
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
 */

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

#define F_1_175_MINUS_1_961  (F_1_175 - F_1_961)
#define F_1_175_MINUS_0_390  (F_1_175 - F_0_390)
#define F_0_541_MINUS_1_847  (F_0_541 - F_1_847)
#define F_3_072_MINUS_2_562  (F_3_072 - F_2_562)
#define F_0_298_MINUS_0_899  (F_0_298 - F_0_899)
#define F_1_501_MINUS_0_899  (F_1_501 - F_0_899)
#define F_2_053_MINUS_2_562  (F_2_053 - F_2_562)
#define F_0_541_PLUS_0_765   (F_0_541 + F_0_765)


/* Forward declaration of regular and sparse IDCT helper functions */

static INLINE void jsimd_idct_islow_pass1_regular(const int16_t * restrict coef_block,
                                                  const int16_t * restrict quantptr,
                                                  int16_t * restrict workspace);

static INLINE void jsimd_idct_islow_pass1_sparse(const int16_t * restrict coef_block,
                                                 const int16_t * restrict quantptr,
                                                 int16_t * restrict workspace);

static INLINE void jsimd_idct_islow_pass2_regular(int16_t * restrict workspace,
                                                  JSAMPARRAY restrict output_buf,
                                                  JDIMENSION output_col,
                                                  unsigned buf_offset);

static INLINE void jsimd_idct_islow_pass2_sparse(int16_t * restrict workspace,
                                                 JSAMPARRAY restrict output_buf,
                                                 JDIMENSION output_col,
                                                 unsigned buf_offset);


/* Perform dequantization and inverse DCT on one block of coefficients.  For
 * reference, the C implementation (jpeg_idct_slow()) can be found in
 * jidctint.c.
 *
 * Optimization techniques used for fast data access:
 *
 * In each pass, the inverse DCT is computed for the left and right 4x8 halves
 * of the DCT block.  This reduces spilling due to register pressure, and the
 * increased granularity allows for an optimized calculation depending on the
 * values of the DCT coefficients.  Between passes, intermediate data is stored
 * in 4x8 workspace buffers.
 *
 * Register pressure is significantly higher on Helium than Neon as we have
 * half the register space, and can't use short vectors. However we use the
 * long vector to perform the inter-pass 4x8->8x4 transpose with one vst4q.
 *
 *  int32x4_t x 8              int16x8x4_t
 *
 *  0000   4444                 40404040                          01234567
 *  1111   5555 -> TB pack ->   51515151     ->   VST4.16    ->   01234567
 *  2222   6666                 62626262                          01234567
 *  3333   7777                 73737373                          01234567
 *
 * The final transpose needs 8 bytes of output - we currently do that
 * by double packing to 0123012301230123 and 4567456745674567, then using
 * two 32-bit scatter stores across four output rows.
 *
 * Optimization techniques used to accelerate the inverse DCT calculation:
 *
 * In a DCT coefficient block, the coefficients are increasingly likely to be 0
 * as you move diagonally from top left to bottom right.  If whole rows of
 * coefficients are 0, then the inverse DCT calculation can be simplified.  On
 * the first pass of the inverse DCT, we test for three special cases before
 * defaulting to a full "regular" inverse DCT:
 *
 * 1) Coefficients in rows 4-7 are all zero.  In this case, we perform a
 *    "sparse" simplified inverse DCT on rows 0-3.
 * 2) AC coefficients (rows 1-7) are all zero.  In this case, the inverse DCT
 *    result is equal to the dequantized DC coefficients.
 * 3) AC and DC coefficients are all zero.  In this case, the inverse DCT
 *    result is all zero.  For the left 4x8 half, this is handled identically
 *    to Case 2 above.  For the right 4x8 half, we do no work and signal that
 *    the "sparse" algorithm is required for the second pass.
 *
 * In the second pass, only a single special case is tested: whether the AC and
 * DC coefficients were all zero in the right 4x8 block during the first pass
 * (refer to Case 3 above.)  If this is the case, then a "sparse" variant of
 * the second pass is performed for both the left and right halves of the DCT
 * block.  (The transposition after the first pass means that the right 4x8
 * block during the first pass becomes rows 4-7 during the second pass.)
 */

void jsimd_idct_islow_helium(void * restrict dct_table, JCOEFPTR restrict coef_block,
                             JSAMPARRAY restrict output_buf, JDIMENSION output_col)
{
#define quant_s32(x, y) vldrhq_s32(quantptr + (y) * DCTSIZE + (x))
#define quant_s16(y) vldrhq_s16(quantptr + (y) * DCTSIZE)
#define row_s32(x, y) vldrhq_s32(coef_block + (y) * DCTSIZE + (x))
#define row_s16(y) vldrhq_s16(coef_block + (y) * DCTSIZE)

  const ISLOW_MULT_TYPE *quantptr = dct_table;

  int16_t workspace[DCTSIZE2];

  /* Construct bitmap (for whole 8x8) to test if DCT coefficients are 0. */
  int16x8_t bitmap = vorrq_s16(row_s16(7), row_s16(6));
  bitmap = vorrq_s16(bitmap, row_s16(5));
  bitmap = vorrq_s16(bitmap, row_s16(4));
  mve_pred16_t bitmap_rows_4567 = vcmpneq_n_s16(bitmap, 0);

  bitmap = vorrq_s16(row_s16(3), row_s16(2));
  bitmap = vorrq_s16(bitmap, row_s16(1));
  mve_pred16_t ac_bitmap = bitmap_rows_4567 | vcmpneq_n_s16(bitmap, 0);
  mve_pred16_t ac_dc_bitmap = ac_bitmap | vcmpneq_n_s16(row_s16(0), 0);

  //bitmap_rows_4567 = ac_bitmap = ac_dc_bitmap = 0xffff;

  /* Compute IDCT first pass on left 4x8 coefficient block. */

  /* If left coefficients 4-7 are all zero */
  if ((bitmap_rows_4567 & 0x00FF) == 0) {
    /* If left AC coefficients (1-7) are all zero */
    if ((ac_bitmap & 0x00FF) == 0) {
      int32x4_t dcval_s32 = vshlq_n_s32(vmulq_s32(row_s32(0, 0), quant_s32(0, 0)), PASS1_BITS);
      int16x8_t dcval = vmovntq_s32(vreinterpretq_s16_s32(dcval_s32), dcval_s32);
      int16x8x4_t quadrant = { { dcval, dcval, dcval, dcval } };
      /* Store 4x8 block to workspace, transposing in the process. */
      vst4q_s16(workspace, quadrant);
    } else {
      jsimd_idct_islow_pass1_sparse(coef_block, quantptr, workspace);
    }
  } else {
    jsimd_idct_islow_pass1_regular(coef_block, quantptr, workspace);
  }

  /* Compute IDCT first pass on right 4x8 coefficient block. */

  /* If right AC coefficients (1-7) are all zero */
  if ((ac_bitmap & 0xFF00) == 0) {
    /* If we have right DC coefficients (if not, the sparse second pass, won't read this) */
    if ((ac_dc_bitmap & 0xFF00) != 0) {
        int32x4_t dcval_s32 = vshlq_n_s32(vmulq_s32(row_s32(4, 0), quant_s32(4, 0)), PASS1_BITS);
        int16x8_t dcval = vmovntq_s32(vreinterpretq_s16_s32(dcval_s32), dcval_s32);
        int16x8x4_t quadrant = { { dcval, dcval, dcval, dcval } };
        /* Store 4x8 block to workspace, transposing in the process. */
        vst4q_s16(workspace + 4 * DCTSIZE, quadrant);
    }
  } else {
    /* If right coefficients 4-7 are all zero */
    if ((bitmap_rows_4567 & 0xFF00) == 0) {
      jsimd_idct_islow_pass1_sparse(coef_block + 4, quantptr + 4,
                                    workspace + 4 * DCTSIZE);
    } else {
      jsimd_idct_islow_pass1_regular(coef_block + 4, quantptr + 4,
                                     workspace + 4 * DCTSIZE);
    }
  }

  /* Second pass: compute IDCT on rows in workspace. */

  /* If all coefficients in right 4x8 block are 0, use "sparse" second pass. */
  if ((ac_dc_bitmap & 0xFF00) == 0) {
    jsimd_idct_islow_pass2_sparse(workspace + 0, output_buf, output_col, 0);
    jsimd_idct_islow_pass2_sparse(workspace + 4, output_buf, output_col, 4);
  } else {
    jsimd_idct_islow_pass2_regular(workspace + 0, output_buf, output_col, 0);
    jsimd_idct_islow_pass2_regular(workspace + 4, output_buf, output_col, 4);
  }
}


/* Perform dequantization and the first pass of the accurate inverse DCT on a
 * 4x8 block of coefficients.  (To process the full 8x8 DCT block, this
 * function-- or some other optimized variant-- needs to be called for both the
 * left and right 4x8 blocks.)
 *
 * This "regular" version assumes that no optimization can be made to the IDCT
 * calculation, since no useful set of AC coefficients is all 0.
 *
 * The original C implementation of the accurate IDCT (jpeg_idct_slow()) can be
 * found in jidctint.c.  Algorithmic changes made here are documented inline.
 */

static INLINE void jsimd_idct_islow_pass1_regular(const int16_t * restrict coef_block,
                                                  const int16_t * restrict quantptr,
                                                  int16_t * restrict workspace)
{
  /* Even part */
  int32x4_t z2_0 = vmulq_s32(row_s32(0, 2), quant_s32(0, 2));
  int32x4_t z3_0 = vmulq_s32(row_s32(0, 6), quant_s32(0, 6));

  int32x4_t tmp2 = vmulq_n_s32(z2_0, F_0_541);
  int32x4_t tmp3 = vmulq_n_s32(z2_0, F_0_541_PLUS_0_765);
  tmp2 = vmlaq_n_s32(tmp2, z3_0, F_0_541_MINUS_1_847);
  tmp3 = vmlaq_n_s32(tmp3, z3_0, F_0_541);

  z2_0 = vmulq_s32(row_s32(0, 0), quant_s32(0, 0));
  z3_0 = vmulq_s32(row_s32(0, 4), quant_s32(0, 4));

  int32x4_t tmp0 = vshlq_n_s32(vaddq_s32(z2_0, z3_0), CONST_BITS);
  int32x4_t tmp1 = vshlq_n_s32(vsubq_s32(z2_0, z3_0), CONST_BITS);

  int32x4_t tmp10 = vaddq_s32(tmp0, tmp3);
  int32x4_t tmp13 = vsubq_s32(tmp0, tmp3);
  int32x4_t tmp11 = vaddq_s32(tmp1, tmp2);
  int32x4_t tmp12 = vsubq_s32(tmp1, tmp2);

  /* Odd part */
  int32x4_t tmp0_1 = vmulq_s32(row_s32(0, 7), quant_s32(0, 7));
  int32x4_t tmp1_1 = vmulq_s32(row_s32(0, 5), quant_s32(0, 5));
  int32x4_t tmp2_1 = vmulq_s32(row_s32(0, 3), quant_s32(0, 3));
  int32x4_t tmp3_1 = vmulq_s32(row_s32(0, 1), quant_s32(0, 1));

  z3_0 = vaddq_s32(tmp0_1, tmp2_1);
  int32x4_t z4_0 = vaddq_s32(tmp1_1, tmp3_1);

  /* Implementation as per jpeg_idct_islow() in jidctint.c:
   *   z5 = (z3 + z4) * 1.175875602;
   *   z3 = z3 * -1.961570560;  z4 = z4 * -0.390180644;
   *   z3 += z5;  z4 += z5;
   *
   * This implementation:
   *   z3 = z3 * (1.175875602 - 1.961570560) + z4 * 1.175875602;
   *   z4 = z3 * 1.175875602 + z4 * (1.175875602 - 0.390180644);
   */

  int32x4_t z3 = vmulq_n_s32(z3_0, F_1_175_MINUS_1_961);
  int32x4_t z4 = vmulq_n_s32(z3_0, F_1_175);
  z3 = vmlaq_n_s32(z3, z4_0, F_1_175);
  z4 = vmlaq_n_s32(z4, z4_0, F_1_175_MINUS_0_390);

  /* Implementation as per jpeg_idct_islow() in jidctint.c:
   *   z1 = tmp0 + tmp3;  z2 = tmp1 + tmp2;
   *   tmp0 = tmp0 * 0.298631336;  tmp1 = tmp1 * 2.053119869;
   *   tmp2 = tmp2 * 3.072711026;  tmp3 = tmp3 * 1.501321110;
   *   z1 = z1 * -0.899976223;  z2 = z2 * -2.562915447;
   *   tmp0 += z1 + z3;  tmp1 += z2 + z4;
   *   tmp2 += z2 + z3;  tmp3 += z1 + z4;
   *
   * This implementation:
   *   tmp0 = tmp0 * (0.298631336 - 0.899976223) + tmp3 * -0.899976223;
   *   tmp1 = tmp1 * (2.053119869 - 2.562915447) + tmp2 * -2.562915447;
   *   tmp2 = tmp1 * -2.562915447 + tmp2 * (3.072711026 - 2.562915447);
   *   tmp3 = tmp0 * -0.899976223 + tmp3 * (1.501321110 - 0.899976223);
   *   tmp0 += z3;  tmp1 += z4;
   *   tmp2 += z3;  tmp3 += z4;
   */

  tmp0 = vmulq_n_s32(tmp0_1, F_0_298_MINUS_0_899);
  tmp1 = vmulq_n_s32(tmp1_1, F_2_053_MINUS_2_562);
  tmp2 = vmulq_n_s32(tmp2_1, F_3_072_MINUS_2_562);
  tmp3 = vmulq_n_s32(tmp3_1, F_1_501_MINUS_0_899);

  tmp0 = vmlaq_n_s32(tmp0, tmp3_1, -F_0_899);
  tmp1 = vmlaq_n_s32(tmp1, tmp2_1, -F_2_562);
  tmp2 = vmlaq_n_s32(tmp2, tmp1_1, -F_2_562);
  tmp3 = vmlaq_n_s32(tmp3, tmp0_1, -F_0_899);

  tmp0 = vaddq_s32(tmp0, z3);
  tmp1 = vaddq_s32(tmp1, z4);
  tmp2 = vaddq_s32(tmp2, z3);
  tmp3 = vaddq_s32(tmp3, z4);

  /* Final output stage: descale and pack */
  int16x8x4_t rows_04_15_26_37 = { {
    vrshrnq_n_s32(vaddq_s32(tmp10, tmp3), vsubq_s32(tmp13, tmp0), DESCALE_P1),
    vrshrnq_n_s32(vaddq_s32(tmp11, tmp2), vsubq_s32(tmp12, tmp1), DESCALE_P1),
    vrshrnq_n_s32(vaddq_s32(tmp12, tmp1), vsubq_s32(tmp11, tmp2), DESCALE_P1),
    vrshrnq_n_s32(vaddq_s32(tmp13, tmp0), vsubq_s32(tmp10, tmp3), DESCALE_P1)
  } };

  /* Store 8x4 result to the intermediate workspace, ready for the second pass.
   * (VST4 transposes the blocks.  We need to operate on rows in the next
   * pass.)
   */
  vst4q_s16(workspace, rows_04_15_26_37);
}


/* Perform dequantization and the first pass of the accurate inverse DCT on a
 * 4x8 block of coefficients.
 *
 * This "sparse" version assumes that the AC coefficients in rows 4-7 are all
 * 0.  This simplifies the IDCT calculation, accelerating overall performance.
 */

static INLINE void jsimd_idct_islow_pass1_sparse(const int16_t * restrict coef_block,
                                                 const int16_t * restrict quantptr,
                                                 int16_t * restrict workspace)
{
  /* Even part (z3 is all 0) */
  int32x4_t z2 = vmulq_s32(row_s32(0, 2), quant_s32(0, 2));

  int32x4_t tmp2 = vmulq_n_s32(z2, F_0_541);
  int32x4_t tmp3 = vmulq_n_s32(z2, F_0_541_PLUS_0_765);

  z2 = vmulq_s32(row_s32(0, 0), quant_s32(0, 0));
  int32x4_t tmp0 = vshlq_n_s32(z2, CONST_BITS);
  int32x4_t tmp1 = vshlq_n_s32(z2, CONST_BITS);

  int32x4_t tmp10 = vaddq_s32(tmp0, tmp3);
  int32x4_t tmp13 = vsubq_s32(tmp0, tmp3);
  int32x4_t tmp11 = vaddq_s32(tmp1, tmp2);
  int32x4_t tmp12 = vsubq_s32(tmp1, tmp2);

  /* Odd part (tmp0 and tmp1 are both all 0) */
  int32x4_t tmp2_0 = vmulq_s32(row_s32(0, 3), quant_s32(0, 3));
  int32x4_t tmp3_0 = vmulq_s32(row_s32(0, 1), quant_s32(0, 1));

  int32x4_t z3_0 = tmp2_0;
  int32x4_t z4_0 = tmp3_0;

  int32x4_t z3 = vmulq_n_s32(z3_0, F_1_175_MINUS_1_961);
  int32x4_t z4 = vmulq_n_s32(z3_0, F_1_175);
  z3 = vmlaq_n_s32(z3, z4_0, F_1_175);
  z4 = vmlaq_n_s32(z4, z4_0, F_1_175_MINUS_0_390);

  tmp0 = vmlaq_n_s32(z3, tmp3_0, -F_0_899);
  tmp1 = vmlaq_n_s32(z4, tmp2_0, -F_2_562);
  tmp2 = vmlaq_n_s32(z3, tmp2_0, F_3_072_MINUS_2_562);
  tmp3 = vmlaq_n_s32(z4, tmp3_0, F_1_501_MINUS_0_899);

  /* Final output stage: descale and pack */
  int16x8x4_t rows_04_15_26_37 = { {
    vrshrnq_n_s32(vaddq_s32(tmp10, tmp3), vsubq_s32(tmp13, tmp0), DESCALE_P1),
    vrshrnq_n_s32(vaddq_s32(tmp11, tmp2), vsubq_s32(tmp12, tmp1), DESCALE_P1),
    vrshrnq_n_s32(vaddq_s32(tmp12, tmp1), vsubq_s32(tmp11, tmp2), DESCALE_P1),
    vrshrnq_n_s32(vaddq_s32(tmp13, tmp0), vsubq_s32(tmp10, tmp3), DESCALE_P1)
  } };

  /* Store 8x4 result to the intermediate workspace, ready for the second pass.
   * (VST4 transposes the blocks.  We need to operate on rows in the next
   * pass.)
   */
  vst4q_s16(workspace, rows_04_15_26_37);
}


/* Perform the second pass of the accurate inverse DCT on a 4x8 block of
 * coefficients.  (To process the full 8x8 DCT block, this function-- or some
 * other optimized variant-- needs to be called for both the right and left 4x8
 * blocks.)
 *
 * This "regular" version assumes that no optimization can be made to the IDCT
 * calculation, since no useful set of coefficient values are all 0 after the
 * first pass.
 *
 * Again, the original C implementation of the accurate IDCT (jpeg_idct_slow())
 * can be found in jidctint.c.  Algorithmic changes made here are documented
 * inline.
 */

static INLINE void jsimd_idct_islow_pass2_regular(int16_t * restrict workspace,
                                                  JSAMPARRAY restrict output_buf,
                                                  JDIMENSION output_col,
                                                  unsigned buf_offset)
{
  /* Even part */
  int32x4_t z2_0 = vldrhq_s32(workspace + 2 * DCTSIZE);
  int32x4_t z3_0 = vldrhq_s32(workspace + 6 * DCTSIZE);

  int32x4_t tmp2 = vmulq_n_s32(z2_0, F_0_541);
  int32x4_t tmp3 = vmulq_n_s32(z2_0, F_0_541_PLUS_0_765);
  tmp2 = vmlaq_n_s32(tmp2, z3_0, F_0_541_MINUS_1_847);
  tmp3 = vmlaq_n_s32(tmp3, z3_0, F_0_541);

  z2_0 = vldrhq_s32(workspace + 0 * DCTSIZE);
  z3_0 = vldrhq_s32(workspace + 4 * DCTSIZE);

  int32x4_t tmp0 = vshlq_n_s32(vaddq_s32(z2_0, z3_0), CONST_BITS);
  int32x4_t tmp1 = vshlq_n_s32(vsubq_s32(z2_0, z3_0), CONST_BITS);

  int32x4_t tmp10 = vaddq_s32(tmp0, tmp3);
  int32x4_t tmp13 = vsubq_s32(tmp0, tmp3);
  int32x4_t tmp11 = vaddq_s32(tmp1, tmp2);
  int32x4_t tmp12 = vsubq_s32(tmp1, tmp2);

  /* Odd part */
  int32x4_t tmp0_1 = vldrhq_s32(workspace + 7 * DCTSIZE);
  int32x4_t tmp1_1 = vldrhq_s32(workspace + 5 * DCTSIZE);
  int32x4_t tmp2_1 = vldrhq_s32(workspace + 3 * DCTSIZE);
  int32x4_t tmp3_1 = vldrhq_s32(workspace + 1 * DCTSIZE);

  z3_0 = vaddq_s32(tmp0_1, tmp2_1);
  int32x4_t z4_0 = vaddq_s32(tmp1_1, tmp3_1);

  /* Implementation as per jpeg_idct_islow() in jidctint.c:
   *   z5 = (z3 + z4) * 1.175875602;
   *   z3 = z3 * -1.961570560;  z4 = z4 * -0.390180644;
   *   z3 += z5;  z4 += z5;
   *
   * This implementation:
   *   z3 = z3 * (1.175875602 - 1.961570560) + z4 * 1.175875602;
   *   z4 = z3 * 1.175875602 + z4 * (1.175875602 - 0.390180644);
   */

  int32x4_t z3 = vmulq_n_s32(z3_0, F_1_175_MINUS_1_961);
  int32x4_t z4 = vmulq_n_s32(z3_0, F_1_175);
  z3 = vmlaq_n_s32(z3, z4_0, F_1_175);
  z4 = vmlaq_n_s32(z4, z4_0, F_1_175_MINUS_0_390);

  /* Implementation as per jpeg_idct_islow() in jidctint.c:
   *   z1 = tmp0 + tmp3;  z2 = tmp1 + tmp2;
   *   tmp0 = tmp0 * 0.298631336;  tmp1 = tmp1 * 2.053119869;
   *   tmp2 = tmp2 * 3.072711026;  tmp3 = tmp3 * 1.501321110;
   *   z1 = z1 * -0.899976223;  z2 = z2 * -2.562915447;
   *   tmp0 += z1 + z3;  tmp1 += z2 + z4;
   *   tmp2 += z2 + z3;  tmp3 += z1 + z4;
   *
   * This implementation:
   *   tmp0 = tmp0 * (0.298631336 - 0.899976223) + tmp3 * -0.899976223;
   *   tmp1 = tmp1 * (2.053119869 - 2.562915447) + tmp2 * -2.562915447;
   *   tmp2 = tmp1 * -2.562915447 + tmp2 * (3.072711026 - 2.562915447);
   *   tmp3 = tmp0 * -0.899976223 + tmp3 * (1.501321110 - 0.899976223);
   *   tmp0 += z3;  tmp1 += z4;
   *   tmp2 += z3;  tmp3 += z4;
   */
  tmp0 = vmulq_n_s32(tmp0_1, F_0_298_MINUS_0_899);
  tmp1 = vmulq_n_s32(tmp1_1, F_2_053_MINUS_2_562);
  tmp2 = vmulq_n_s32(tmp2_1, F_3_072_MINUS_2_562);
  tmp3 = vmulq_n_s32(tmp3_1, F_1_501_MINUS_0_899);

  tmp0 = vmlaq_n_s32(tmp0, tmp3_1, -F_0_899);
  tmp1 = vmlaq_n_s32(tmp1, tmp2_1, -F_2_562);
  tmp2 = vmlaq_n_s32(tmp2, tmp1_1, -F_2_562);
  tmp3 = vmlaq_n_s32(tmp3, tmp0_1, -F_0_899);

  tmp0 = vaddq_s32(tmp0, z3);
  tmp1 = vaddq_s32(tmp1, z4);
  tmp2 = vaddq_s32(tmp2, z3);
  tmp3 = vaddq_s32(tmp3, z4);

  /* Final output stage: descale and narrow to 16-bit, and pack */
  int16x8_t cols_02_s16 = vshrnq_16_s32(vaddq_s32(tmp10, tmp3),
                                        vaddq_s32(tmp12, tmp1));
  int16x8_t cols_13_s16 = vshrnq_16_s32(vaddq_s32(tmp11, tmp2),
                                        vaddq_s32(tmp13, tmp0));
  int16x8_t cols_46_s16 = vshrnq_16_s32(vsubq_s32(tmp13, tmp0),
                                        vsubq_s32(tmp11, tmp2));
  int16x8_t cols_57_s16 = vshrnq_16_s32(vsubq_s32(tmp12, tmp1),
                                        vsubq_s32(tmp10, tmp3));
  /* Descale and narrow to 8-bit, clamping to range [-128-127], and pack */
  int8x16_t cols_0123_s8 = vqrshrnq_n_s16(cols_02_s16, cols_13_s16, DESCALE_P2 - 16);
  int8x16_t cols_4567_s8 = vqrshrnq_n_s16(cols_46_s16, cols_57_s16, DESCALE_P2 - 16);
  /* Shift to range [0-255]. */
  uint8x16_t cols_0123_u8 = vaddq_n_u8(vreinterpretq_u8_s8(cols_0123_s8), CENTERJSAMPLE);
  uint8x16_t cols_4567_u8 = vaddq_n_u8(vreinterpretq_u8_s8(cols_4567_s8), CENTERJSAMPLE);
  /* Reinterpret as 32-bit units for final transpose */
  uint32x4_t cols_0123_u32 = vreinterpretq_u32_u8(cols_0123_u8);
  uint32x4_t cols_4567_u32 = vreinterpretq_u32_u8(cols_4567_u8);

  /* Last transposition through scatter stores */
  uint32x4_t outptrs = vld1q_u32((uint32_t *) (output_buf + buf_offset));
  outptrs = vaddq_n_u32(outptrs, output_col);
  vstrwq_scatter_base_u32(outptrs, 0, cols_0123_u32);
  vstrwq_scatter_base_u32(outptrs, 4, cols_4567_u32);
}


/* Performs the second pass of the accurate inverse DCT on a 4x8 block
 * of coefficients.
 *
 * This "sparse" version assumes that the coefficient values (after the first
 * pass) in rows 4-7 are all 0.  This simplifies the IDCT calculation,
 * accelerating overall performance.
 */

static INLINE void jsimd_idct_islow_pass2_sparse(int16_t * restrict workspace,
                                                 JSAMPARRAY restrict output_buf,
                                                 JDIMENSION output_col,
                                                 unsigned buf_offset)
{
  /* Even part (z3 is all 0) */
  int32x4_t z2_0 = vldrhq_s32(workspace + 2 * DCTSIZE);

  int32x4_t tmp2 = vmulq_n_s32(z2_0, F_0_541);
  int32x4_t tmp3 = vmulq_n_s32(z2_0, F_0_541_PLUS_0_765);

  z2_0 = vldrhq_s32(workspace + 0 * DCTSIZE);
  int32x4_t tmp0 = vshlq_n_s32(z2_0, CONST_BITS);
  int32x4_t tmp1 = vshlq_n_s32(z2_0, CONST_BITS);

  int32x4_t tmp10 = vaddq_s32(tmp0, tmp3);
  int32x4_t tmp13 = vsubq_s32(tmp0, tmp3);
  int32x4_t tmp11 = vaddq_s32(tmp1, tmp2);
  int32x4_t tmp12 = vsubq_s32(tmp1, tmp2);

  /* Odd part (tmp0 and tmp1 are both all 0) */
  int32x4_t tmp2_1 = vldrhq_s32(workspace + 3 * DCTSIZE);
  int32x4_t tmp3_1 = vldrhq_s32(workspace + 1 * DCTSIZE);

  int32x4_t z3_0 = tmp2_1;
  int32x4_t z4_0 = tmp3_1;

  int32x4_t z3 = vmulq_n_s32(z3_0, F_1_175_MINUS_1_961);
  z3 = vmlaq_n_s32(z3, z4_0, F_1_175);
  int32x4_t z4 = vmulq_n_s32(z3_0, F_1_175);
  z4 = vmlaq_n_s32(z4, z4_0, F_1_175_MINUS_0_390);

  tmp0 = vmlaq_n_s32(z3, tmp3_1, -F_0_899);
  tmp1 = vmlaq_n_s32(z4, tmp2_1, -F_2_562);
  tmp2 = vmlaq_n_s32(z3, tmp2_1, F_3_072_MINUS_2_562);
  tmp3 = vmlaq_n_s32(z4, tmp3_1, F_1_501_MINUS_0_899);

  /* Final output stage: descale and narrow to 16-bit, and pack */
  int16x8_t cols_02_s16 = vshrnq_16_s32(vaddq_s32(tmp10, tmp3),
                                        vaddq_s32(tmp12, tmp1));
  int16x8_t cols_13_s16 = vshrnq_16_s32(vaddq_s32(tmp11, tmp2),
                                        vaddq_s32(tmp13, tmp0));
  int16x8_t cols_46_s16 = vshrnq_16_s32(vsubq_s32(tmp13, tmp0),
                                        vsubq_s32(tmp11, tmp2));
  int16x8_t cols_57_s16 = vshrnq_16_s32(vsubq_s32(tmp12, tmp1),
                                        vsubq_s32(tmp10, tmp3));
  /* Descale and narrow to 8-bit, clamping to range [-128-127], and pack */
  int8x16_t cols_0123_s8 = vqrshrnq_n_s16(cols_02_s16, cols_13_s16, DESCALE_P2 - 16);
  int8x16_t cols_4567_s8 = vqrshrnq_n_s16(cols_46_s16, cols_57_s16, DESCALE_P2 - 16);
  /* Shift to range [0-255]. */
  uint8x16_t cols_0123_u8 = vaddq_n_u8(vreinterpretq_u8_s8(cols_0123_s8), CENTERJSAMPLE);
  uint8x16_t cols_4567_u8 = vaddq_n_u8(vreinterpretq_u8_s8(cols_4567_s8), CENTERJSAMPLE);
  /* Reinterpret as 32-bit units for final transpose */
  uint32x4_t cols_0123_u32 = vreinterpretq_u32_u8(cols_0123_u8);
  uint32x4_t cols_4567_u32 = vreinterpretq_u32_u8(cols_4567_u8);

  /* Last transposition through scatter stores */
  uint32x4_t outptrs = vld1q_u32((uint32_t *) (output_buf + buf_offset));
  outptrs = vaddq_n_u32(outptrs, output_col);
  vstrwq_scatter_base_u32(outptrs, 0, cols_0123_u32);
  vstrwq_scatter_base_u32(outptrs, 4, cols_4567_u32);
}
