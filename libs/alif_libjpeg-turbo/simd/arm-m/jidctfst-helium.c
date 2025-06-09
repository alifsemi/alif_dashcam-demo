/*
 * jidctfst-helium.c - fast integer IDCT (Arm Helium)
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
#include "helium-helpers.h"

#include <arm_mve.h>


/* jsimd_idct_ifast_helium() performs dequantization and a fast, not so accurate
 * inverse DCT (Discrete Cosine Transform) on one block of coefficients.  It
 * uses the same calculations and produces exactly the same output as IJG's
 * original jpeg_idct_ifast() function, which can be found in jidctfst.c.
 *
 * Scaled integer constants are used to avoid floating-point arithmetic:
 *    0.082392200 =  2688 * 2^-15
 *    0.414213562 = 13568 * 2^-15
 *    0.847759065 = 27776 * 2^-15
 *    0.613125930 = 20096 * 2^-15
 *
 * See jidctfst.c for further details of the IDCT algorithm.  Where possible,
 * the variable names and comments here in jsimd_idct_ifast_neon() match up
 * with those in jpeg_idct_ifast().
 */

#define PASS1_BITS  2

#define F_0_082  2688
#define F_0_414  13568
#define F_0_847  27776
#define F_0_613  20096


/* We're (ab)using the VSTRD.64 instruction by putting pointers in
 * the vector and offsets in the scalar. Deal with the typing issues
 * here. Also treat address vector as 32-bit, as it's more natural;
 * only lanes 0 and 2 are used.
 */
static INLINE void
vstrdq_scatter_r_offset_u64(uint32x4_t base, uint32_t offset, uint64x2_t value)
{
  vstrdq_scatter_offset_u64((uint64_t *) offset, vreinterpretq_u64_u32(base), value);
}

void jsimd_idct_ifast_helium(void * restrict dct_table, JCOEFPTR restrict coef_block,
                             JSAMPARRAY restrict output_buf, JDIMENSION output_col)
{
#define quant_s16(y) vldrhq_s16(quantptr + (y) * DCTSIZE)
#define row_s16(y) vldrhq_s16(coef_block + (y) * DCTSIZE)
#define col_s16(y) vldrhq_s16(workspace + (y) * DCTSIZE)

  IFAST_MULT_TYPE *quantptr = dct_table;

  int16_t workspace[DCTSIZE2];

#if 0 // Testing shows this not to be a net win - unikely 56 coefficients are zero */
  /* Construct bitmap to test if all AC coefficients are 0. */
  int16x8_t bitmap = vorrq_s16(row_s16(1), row_s16(2));
  bitmap = vorrq_s16(bitmap, row_s16(3));
  bitmap = vorrq_s16(bitmap, row_s16(4));
  bitmap = vorrq_s16(bitmap, row_s16(5));
  bitmap = vorrq_s16(bitmap, row_s16(6));
  bitmap = vorrq_s16(bitmap, row_s16(7));
  mve_pred16_t ac_bitmap = vcmpneq_n_s16(bitmap, 0);

  if (ac_bitmap == 0) {
    /* All AC coefficients are zero.
     * Compute DC values and duplicate into vectors.
     */
    int16x8_t row0 = vmulq_s16(row_s16(0), quant_s16(0));
    vstrhq_s16(workspace + 0 * DCTSIZE, vdupq_n_s16(vgetq_lane_s16(row0, 0)));
    vstrhq_s16(workspace + 1 * DCTSIZE, vdupq_n_s16(vgetq_lane_s16(row0, 1)));
    vstrhq_s16(workspace + 2 * DCTSIZE, vdupq_n_s16(vgetq_lane_s16(row0, 2)));
    vstrhq_s16(workspace + 3 * DCTSIZE, vdupq_n_s16(vgetq_lane_s16(row0, 3)));
    vstrhq_s16(workspace + 4 * DCTSIZE, vdupq_n_s16(vgetq_lane_s16(row0, 4)));
    vstrhq_s16(workspace + 5 * DCTSIZE, vdupq_n_s16(vgetq_lane_s16(row0, 5)));
    vstrhq_s16(workspace + 6 * DCTSIZE, vdupq_n_s16(vgetq_lane_s16(row0, 6)));
    vstrhq_s16(workspace + 7 * DCTSIZE, vdupq_n_s16(vgetq_lane_s16(row0, 7)));
  } else {
    /* Some AC coefficients are non-zero; full IDCT calculation required. */
#else
  {
#endif

    /* Even part: dequantize DCT coefficients. */
    int16x8_t tmp0 = vmulq_s16(row_s16(0), quant_s16(0));
    int16x8_t tmp1 = vmulq_s16(row_s16(2), quant_s16(2));
    int16x8_t tmp2 = vmulq_s16(row_s16(4), quant_s16(4));
    int16x8_t tmp3 = vmulq_s16(row_s16(6), quant_s16(6));

    int16x8_t tmp10 = vaddq_s16(tmp0, tmp2);   /* phase 3 */
    int16x8_t tmp11 = vsubq_s16(tmp0, tmp2);

    int16x8_t tmp13 = vaddq_s16(tmp1, tmp3);   /* phases 5-3 */
    int16x8_t tmp1_sub_tmp3 = vsubq_s16(tmp1, tmp3);
    int16x8_t tmp12 = vqdmlahq_n_s16(tmp1_sub_tmp3, tmp1_sub_tmp3, F_0_414);
    tmp12 = vsubq_s16(tmp12, tmp13);

    tmp0 = vaddq_s16(tmp10, tmp13);            /* phase 2 */
    tmp3 = vsubq_s16(tmp10, tmp13);
    tmp1 = vaddq_s16(tmp11, tmp12);
    tmp2 = vsubq_s16(tmp11, tmp12);

    /* Odd part: dequantize DCT coefficients. */
    int16x8_t tmp4 = vmulq_s16(row_s16(1), quant_s16(1));
    int16x8_t tmp5 = vmulq_s16(row_s16(3), quant_s16(3));
    int16x8_t tmp6 = vmulq_s16(row_s16(5), quant_s16(5));
    int16x8_t tmp7 = vmulq_s16(row_s16(7), quant_s16(7));

    int16x8_t z13 = vaddq_s16(tmp6, tmp5);     /* phase 6 */
    int16x8_t neg_z10 = vsubq_s16(tmp5, tmp6);
    int16x8_t z11 = vaddq_s16(tmp4, tmp7);
    int16x8_t z12 = vsubq_s16(tmp4, tmp7);

    tmp7 = vaddq_s16(z11, z13);                /* phase 5 */
    int16x8_t z11_sub_z13 = vsubq_s16(z11, z13);
    tmp11 = vqdmlahq_n_s16(z11_sub_z13, z11_sub_z13, F_0_414);

    int16x8_t z10_add_z12 = vsubq_s16(z12, neg_z10);
    int16x8_t z5 = vqdmlahq_n_s16(z10_add_z12, z10_add_z12, F_0_847);
    tmp10 = vqdmlahq_n_s16(z12, z12, F_0_082);
    tmp10 = vsubq_s16(tmp10, z5);
    tmp12 = vqdmlahq_n_s16(neg_z10, neg_z10, F_0_613);
    tmp12 = vaddq_s16(tmp12, neg_z10);
    tmp12 = vaddq_s16(tmp12, z5);

    tmp6 = vsubq_s16(tmp12, tmp7);             /* phase 2 */
    tmp5 = vsubq_s16(tmp11, tmp6);
    tmp4 = vaddq_s16(tmp10, tmp5);

    int16x8_t row0 = vaddq_s16(tmp0, tmp7);
    int16x8_t row7 = vsubq_s16(tmp0, tmp7);
    int16x8_t row1 = vaddq_s16(tmp1, tmp6);
    int16x8_t row6 = vsubq_s16(tmp1, tmp6);
    int16x8_t row2 = vaddq_s16(tmp2, tmp5);
    int16x8_t row5 = vsubq_s16(tmp2, tmp5);
    int16x8_t row4 = vaddq_s16(tmp3, tmp4);
    int16x8_t row3 = vsubq_s16(tmp3, tmp4);

    /* Transpose rows to work on columns in pass 2. */
    const uint16x8_t offsets = vidupq_n_u16(0, DCTSIZE);
    vstrhq_scatter_shifted_offset_s16(workspace + 0, offsets, row0);
    vstrhq_scatter_shifted_offset_s16(workspace + 1, offsets, row1);
    vstrhq_scatter_shifted_offset_s16(workspace + 2, offsets, row2);
    vstrhq_scatter_shifted_offset_s16(workspace + 3, offsets, row3);
    vstrhq_scatter_shifted_offset_s16(workspace + 4, offsets, row4);
    vstrhq_scatter_shifted_offset_s16(workspace + 5, offsets, row5);
    vstrhq_scatter_shifted_offset_s16(workspace + 6, offsets, row6);
    vstrhq_scatter_shifted_offset_s16(workspace + 7, offsets, row7);
  }

  /* 1-D IDCT, pass 2 */

  /* Even part */
  int16x8_t tmp10 = vaddq_s16(col_s16(0), col_s16(4));
  int16x8_t tmp11 = vsubq_s16(col_s16(0), col_s16(4));

  int16x8_t tmp13 = vaddq_s16(col_s16(2), col_s16(6));
  int16x8_t col2_sub_col6 = vsubq_s16(col_s16(2), col_s16(6));
  int16x8_t tmp12 = vqdmlahq_n_s16(col2_sub_col6, col2_sub_col6, F_0_414);
  tmp12 = vsubq_s16(tmp12, tmp13);

  int16x8_t tmp0 = vaddq_s16(tmp10, tmp13);
  int16x8_t tmp3 = vsubq_s16(tmp10, tmp13);
  int16x8_t tmp1 = vaddq_s16(tmp11, tmp12);
  int16x8_t tmp2 = vsubq_s16(tmp11, tmp12);

  /* Odd part */
  int16x8_t z13 = vaddq_s16(col_s16(5), col_s16(3));
  int16x8_t neg_z10 = vsubq_s16(col_s16(3), col_s16(5));
  int16x8_t z11 = vaddq_s16(col_s16(1), col_s16(7));
  int16x8_t z12 = vsubq_s16(col_s16(1), col_s16(7));

  int16x8_t tmp7 = vaddq_s16(z11, z13);      /* phase 5 */
  int16x8_t z11_sub_z13 = vsubq_s16(z11, z13);
  tmp11 = vqdmlahq_n_s16(z11_sub_z13, z11_sub_z13, F_0_414);

  int16x8_t z10_add_z12 = vsubq_s16(z12, neg_z10);
  int16x8_t z5 = vqdmlahq_n_s16(z10_add_z12, z10_add_z12, F_0_847);
  tmp10 = vqdmlahq_n_s16(z12, z12, F_0_082);
  tmp10 = vsubq_s16(tmp10, z5);
  tmp12 = vqdmlahq_n_s16(neg_z10, neg_z10, F_0_613);
  tmp12 = vaddq_s16(tmp12, neg_z10);
  tmp12 = vaddq_s16(tmp12, z5);

  int16x8_t tmp6 = vsubq_s16(tmp12, tmp7);   /* phase 2 */
  int16x8_t tmp5 = vsubq_s16(tmp11, tmp6);
  int16x8_t tmp4 = vaddq_s16(tmp10, tmp5);

  int16x8_t col0 = vaddq_s16(tmp0, tmp7);
  int16x8_t col7 = vsubq_s16(tmp0, tmp7);
  int16x8_t col1 = vaddq_s16(tmp1, tmp6);
  int16x8_t col6 = vsubq_s16(tmp1, tmp6);
  int16x8_t col2 = vaddq_s16(tmp2, tmp5);
  int16x8_t col5 = vsubq_s16(tmp2, tmp5);
  int16x8_t col4 = vaddq_s16(tmp3, tmp4);
  int16x8_t col3 = vsubq_s16(tmp3, tmp4);

  /* Scale down by a factor of 8, narrowing to 8-bit, clamping to range [-128-127]. */
  int8x16_t cols_01_s8 = vqshrnq_n_s16(col0, col1, PASS1_BITS + 3);
  int8x16_t cols_23_s8 = vqshrnq_n_s16(col2, col3, PASS1_BITS + 3);
  int8x16_t cols_45_s8 = vqshrnq_n_s16(col4, col5, PASS1_BITS + 3);
  int8x16_t cols_67_s8 = vqshrnq_n_s16(col6, col7, PASS1_BITS + 3);

  /* Shift to range [0-255]. */
  uint8x16_t cols_01 = vaddq_n_u8(vreinterpretq_u8_s8(cols_01_s8), CENTERJSAMPLE);
  uint8x16_t cols_23 = vaddq_n_u8(vreinterpretq_u8_s8(cols_23_s8), CENTERJSAMPLE);
  uint8x16_t cols_45 = vaddq_n_u8(vreinterpretq_u8_s8(cols_45_s8), CENTERJSAMPLE);
  uint8x16_t cols_67 = vaddq_n_u8(vreinterpretq_u8_s8(cols_67_s8), CENTERJSAMPLE);

  /* Transpose into our output buffer to prepare for final store */
  int16x8x4_t cols_01_23_45_67 = { {
    vreinterpretq_s16_u8(cols_01),
    vreinterpretq_s16_u8(cols_23),
    vreinterpretq_s16_u8(cols_45),
    vreinterpretq_s16_u8(cols_67)
  } };
  vst4q_s16(workspace, cols_01_23_45_67);

  /* Reload transposed data as 64-bit units */
  uint64x2_t output01 = vreinterpretq_u64_s16(vld1q_s16(workspace + 0 * DCTSIZE / 2));
  uint64x2_t output23 = vreinterpretq_u64_s16(vld1q_s16(workspace + 2 * DCTSIZE / 2));
  uint64x2_t output45 = vreinterpretq_u64_s16(vld1q_s16(workspace + 4 * DCTSIZE / 2));
  uint64x2_t output67 = vreinterpretq_u64_s16(vld1q_s16(workspace + 6 * DCTSIZE / 2));
#if 1
  /* Load the 8 row pointers into vectors (using VMOV Qd[2], Qd[0]) */
  uint32x4_t output_bufs_01 = vuninitializedq_u32();
  uint32x4_t output_bufs_23 = vuninitializedq_u32();
  uint32x4_t output_bufs_45 = vuninitializedq_u32();
  uint32x4_t output_bufs_67 = vuninitializedq_u32();
  output_bufs_01 = vsetq_lane_u32((uint32_t) output_buf[0], output_bufs_01, 0);
  output_bufs_01 = vsetq_lane_u32((uint32_t) output_buf[1], output_bufs_01, 2);
  output_bufs_23 = vsetq_lane_u32((uint32_t) output_buf[2], output_bufs_23, 0);
  output_bufs_23 = vsetq_lane_u32((uint32_t) output_buf[3], output_bufs_23, 2);
  output_bufs_45 = vsetq_lane_u32((uint32_t) output_buf[4], output_bufs_45, 0);
  output_bufs_45 = vsetq_lane_u32((uint32_t) output_buf[5], output_bufs_45, 2);
  output_bufs_67 = vsetq_lane_u32((uint32_t) output_buf[6], output_bufs_67, 0);
  output_bufs_67 = vsetq_lane_u32((uint32_t) output_buf[7], output_bufs_67, 2);
#else
  uint64x2_t offsets = vcreateq_u64(4, 0);
  uint64x2_t output_bufs_01 = vldrdq_gather_offset_u64((const uint64_t *) (output_buf + 0), offsets);
  uint64x2_t output_bufs_23 = vldrdq_gather_offset_u64((const uint64_t *) (output_buf + 2), offsets);
  uint64x2_t output_bufs_45 = vldrdq_gather_offset_u64((const uint64_t *) (output_buf + 4), offsets);
  uint64x2_t output_bufs_67 = vldrdq_gather_offset_u64((const uint64_t *) (output_buf + 6), offsets);
#endif
  /* Scatter the 64-bit units into 8 rows */
  vstrdq_scatter_r_offset_u64(output_bufs_01, output_col, output01);
  vstrdq_scatter_r_offset_u64(output_bufs_23, output_col, output23);
  vstrdq_scatter_r_offset_u64(output_bufs_45, output_col, output45);
  vstrdq_scatter_r_offset_u64(output_bufs_67, output_col, output67);
}
