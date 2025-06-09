/*
 * jidctflt-helium.c - floating-point IDCT (Arm Helium)
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

#if __ARM_FEATURE_MVE & 2

/* Forward declaration of regular and sparse IDCT helper functions */

static INLINE void jsimd_idct_float_pass1_regular(const int16_t * restrict coef_block,
                                                  const float32_t * restrict quantptr,
                                                  float32_t * restrict workspace_1,
                                                  float32_t * restrict workspace_2);

static INLINE void jsimd_idct_float_pass1_sparse(const int16_t * restrict coef_block,
                                                 const float32_t * restrict quantptr,
                                                 float32_t * restrict workspace_1,
                                                 float32_t * restrict workspace_2);

static INLINE void jsimd_idct_float_pass2_regular(float32_t * restrict workspace,
                                                  JSAMPARRAY restrict output_buf,
                                                  JDIMENSION output_col,
                                                  unsigned buf_offset);

static INLINE void jsimd_idct_float_pass2_sparse(float32_t * restrict workspace,
                                                 JSAMPARRAY restrict output_buf,
                                                 JDIMENSION output_col,
                                                 unsigned buf_offset);


/* Perform dequantization and inverse DCT on one block of coefficients.  For
 * reference, the C implementation (jpeg_idct_float()) can be found in
 * jidctflt.c.
 *
 * Optimization techniques used for fast data access:
 *
 * In each pass, the inverse DCT is computed for the left and right 4x8 halves
 * of the DCT block.  This avoids spilling due to register pressure, and the
 * increased granularity allows for an optimized calculation depending on the
 * values of the DCT coefficients.  Between passes, intermediate data is stored
 * in 4x8 workspace buffers.
 *
 * Transposing the 8x8 DCT block after each pass can be achieved by transposing
 * each of the four 4x4 quadrants and swapping quadrants 1 and 2 (refer to the
 * diagram below.)  Swapping quadrants is cheap, since the second pass can just
 * swap the workspace buffer pointers.
 *
 *      +-------+-------+                   +-------+-------+
 *      |       |       |                   |       |       |
 *      |   0   |   1   |                   |   0   |   2   |
 *      |       |       |    transpose      |       |       |
 *      +-------+-------+     ------>       +-------+-------+
 *      |       |       |                   |       |       |
 *      |   2   |   3   |                   |   1   |   3   |
 *      |       |       |                   |       |       |
 *      +-------+-------+                   +-------+-------+
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

void jsimd_idct_float_helium(void * restrict dct_table, JCOEFPTR restrict coef_block,
                             JSAMPARRAY restrict output_buf, JDIMENSION output_col)
{
#define quant_f32(x, y) vldrwq_f32(quantptr + (y) * DCTSIZE + (x))
#define row_s32(x, y) vldrhq_s32(coef_block + (y) * DCTSIZE + (x))
#define row_s16(y) vldrhq_s16(coef_block + (y) * DCTSIZE)

  const FLOAT_MULT_TYPE *quantptr = dct_table;

  float32_t workspace_l[8 * DCTSIZE / 2];
  float32_t workspace_r[8 * DCTSIZE / 2];

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
      float32x4_t dcval = vmulq_f32(vcvtq_n_f32_s32(row_s32(0, 0), 3), quant_f32(0, 0));
      float32x4x4_t quadrant = { { dcval, dcval, dcval, dcval } };
      /* Store 4x4 blocks to workspace, transposing in the process. */
      vst4q_f32(workspace_l, quadrant);
      vst4q_f32(workspace_r, quadrant);
    } else {
      jsimd_idct_float_pass1_sparse(coef_block, quantptr, workspace_l, workspace_r);
    }
  } else {
    jsimd_idct_float_pass1_regular(coef_block, quantptr, workspace_l, workspace_r);
  }

  /* Compute IDCT first pass on right 4x8 coefficient block. */

  /* If right AC coefficients (1-7) are all zero */
  if ((ac_bitmap & 0xFF00) == 0) {
    /* If we have right DC coefficients (if not, the sparse second pass, won't read this) */
    if ((ac_dc_bitmap & 0xFF00) != 0) {

        float32x4_t dcval = vmulq_f32(vcvtq_n_f32_s32(row_s32(4, 0), 3), quant_f32(4, 0));
        float32x4x4_t quadrant = { { dcval, dcval, dcval, dcval } };
        /* Store 4x4 blocks to workspace, transposing in the process. */
        vst4q_f32(workspace_l + 4 * DCTSIZE / 2, quadrant);
        vst4q_f32(workspace_r + 4 * DCTSIZE / 2, quadrant);
    }
  } else {
    /* If right coefficients 4-7 are all zero */
    if ((bitmap_rows_4567 & 0xFF00) == 0) {
      jsimd_idct_float_pass1_sparse(coef_block + 4, quantptr + 4,
                                    workspace_l + 4 * DCTSIZE / 2,
                                    workspace_r + 4 * DCTSIZE / 2);
    } else {
      jsimd_idct_float_pass1_regular(coef_block + 4, quantptr + 4,
                                     workspace_l + 4 * DCTSIZE / 2,
                                     workspace_r + 4 * DCTSIZE / 2);
    }
  }

  /* Second pass: compute IDCT on rows in workspace. */

  /* If all coefficients in right 4x8 block are 0, use "sparse" second pass. */
  if ((ac_dc_bitmap & 0xFF00) == 0) {
    jsimd_idct_float_pass2_sparse(workspace_l, output_buf, output_col, 0);
    jsimd_idct_float_pass2_sparse(workspace_r, output_buf, output_col, 4);
  } else {
    jsimd_idct_float_pass2_regular(workspace_l, output_buf, output_col, 0);
    jsimd_idct_float_pass2_regular(workspace_r, output_buf, output_col, 4);
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
 * The original C implementation of the accurate IDCT (jpeg_idct_float()) can be
 * found in jidctflt.c.  Algorithmic changes made here are documented inline.
 */

static INLINE void jsimd_idct_float_pass1_regular(const int16_t * restrict coef_block,
                                                  const float32_t * restrict quantptr,
                                                  float32_t * restrict workspace_1,
                                                  float32_t * restrict workspace_2)
{
  /* Even part */

  float32x4_t tmp0 = vmulq_f32(vcvtq_n_f32_s32(row_s32(0, 0), 3), quant_f32(0, 0));
  float32x4_t tmp1 = vmulq_f32(vcvtq_n_f32_s32(row_s32(0, 2), 3), quant_f32(0, 2));
  float32x4_t tmp2 = vmulq_f32(vcvtq_n_f32_s32(row_s32(0, 4), 3), quant_f32(0, 4));
  float32x4_t tmp3 = vmulq_f32(vcvtq_n_f32_s32(row_s32(0, 6), 3), quant_f32(0, 6));

  float32x4_t tmp10 = vaddq_f32(tmp0, tmp2);
  float32x4_t tmp11 = vsubq_f32(tmp0, tmp2);

  float32x4_t tmp13 = vaddq_f32(tmp1, tmp3);
  float32x4_t tmp12 = vfmaq_n_f32(tmp13, vsubq_f32(tmp3, tmp1), 1.414213562f);

  tmp0 = vaddq_f32(tmp10, tmp13);
  tmp3 = vsubq_f32(tmp10, tmp13);
  tmp1 = vsubq_f32(tmp11, tmp12);
  tmp2 = vaddq_f32(tmp11, tmp12);

  /* Odd part */

  float32x4_t tmp4 = vmulq_f32(vcvtq_n_f32_s32(row_s32(0, 1), 3), quant_f32(0, 1));
  float32x4_t tmp5 = vmulq_f32(vcvtq_n_f32_s32(row_s32(0, 3), 3), quant_f32(0, 3));
  float32x4_t tmp6 = vmulq_f32(vcvtq_n_f32_s32(row_s32(0, 5), 3), quant_f32(0, 5));
  float32x4_t tmp7 = vmulq_f32(vcvtq_n_f32_s32(row_s32(0, 7), 3), quant_f32(0, 7));

  float32x4_t z13 = vaddq_f32(tmp6, tmp5);
  float32x4_t neg_z10 = vsubq_f32(tmp5, tmp6);
  float32x4_t z11 = vaddq_f32(tmp4, tmp7);
  float32x4_t z12 = vsubq_f32(tmp4, tmp7);

  tmp7 = vaddq_f32(z11, z13);
  tmp11 = vmulq_n_f32(vsubq_f32(z11, z13), 1.414213562f);

  float32x4_t z5 = vmulq_n_f32(vsubq_f32(z12, neg_z10), 1.847759065f);
  tmp10 = vfmaq_n_f32(z5, z12, -1.082392200f);
  tmp12 = vfmaq_n_f32(z5, neg_z10, +2.613125930f);

  tmp6 = vsubq_f32(tmp12, tmp7);        /* phase 2 */
  tmp5 = vsubq_f32(tmp11, tmp6);
  tmp4 = vsubq_f32(tmp10, tmp5);

  /* Final output stage. */
  float32x4x4_t rows_0123 = { {
    vaddq_f32(tmp0, tmp7),
    vaddq_f32(tmp1, tmp6),
    vaddq_f32(tmp2, tmp5),
    vaddq_f32(tmp3, tmp4)
  } };
  float32x4x4_t rows_4567 = { {
    vsubq_f32(tmp3, tmp4),
    vsubq_f32(tmp2, tmp5),
    vsubq_f32(tmp1, tmp6),
    vsubq_f32(tmp0, tmp7)
  } };

  /* Store 4x4 blocks to the intermediate workspace, ready for the second pass.
   * (VST4 transposes the blocks.  We need to operate on rows in the next
   * pass.)
   */
  vst4q_f32(workspace_1, rows_0123);
  vst4q_f32(workspace_2, rows_4567);
}


/* Perform dequantization and the first pass of the accurate inverse DCT on a
 * 4x8 block of coefficients.
 *
 * This "sparse" version assumes that the AC coefficients in rows 4-7 are all
 * 0.  This simplifies the IDCT calculation, accelerating overall performance.
 */

static INLINE void jsimd_idct_float_pass1_sparse(const int16_t * restrict coef_block,
                                                 const float32_t * restrict quantptr,
                                                 float32_t * restrict workspace_1,
                                                 float32_t * restrict workspace_2)
{
  /* Even part */

  float32x4_t tmp0 = vmulq_f32(vcvtq_n_f32_s32(row_s32(0, 0), 3), quant_f32(0, 0));
  float32x4_t tmp1 = vmulq_f32(vcvtq_n_f32_s32(row_s32(0, 2), 3), quant_f32(0, 2));
  float32x4_t tmp2;
  float32x4_t tmp3;

  float32x4_t tmp10 = tmp0;
  float32x4_t tmp11 = tmp0;

  float32x4_t tmp13 = tmp1;
  float32x4_t tmp12 = vmulq_n_f32(tmp1, 0.414213562f);

  tmp0 = vaddq_f32(tmp10, tmp13);
  tmp3 = vsubq_f32(tmp10, tmp13);
  tmp1 = vaddq_f32(tmp11, tmp12);
  tmp2 = vsubq_f32(tmp11, tmp12);

  /* Odd part */

  float32x4_t tmp4 = vmulq_f32(vcvtq_n_f32_s32(row_s32(0, 1), 3), quant_f32(0, 1));
  float32x4_t tmp5 = vmulq_f32(vcvtq_n_f32_s32(row_s32(0, 3), 3), quant_f32(0, 3));

  float32x4_t tmp7 = vaddq_f32(tmp4, tmp5);
  tmp11 = vmulq_n_f32(vsubq_f32(tmp4, tmp5), 1.414213562f);

  float32x4_t z5 = vmulq_n_f32(vsubq_f32(tmp4, tmp5), 1.847759065f);
  tmp10 = vfmaq_n_f32(z5, tmp4, -1.082392200f);
  tmp12 = vfmaq_n_f32(z5, tmp5, +2.613125930f);

  float32x4_t tmp6 = vsubq_f32(tmp12, tmp7);
  tmp5 = vsubq_f32(tmp11, tmp6);
  tmp4 = vsubq_f32(tmp10, tmp5);

  /* Final output stage. */
  float32x4x4_t rows_0123 = { {
    vaddq_f32(tmp0, tmp7),
    vaddq_f32(tmp1, tmp6),
    vaddq_f32(tmp2, tmp5),
    vaddq_f32(tmp3, tmp4)
  } };
  float32x4x4_t rows_4567 = { {
    vsubq_f32(tmp3, tmp4),
    vsubq_f32(tmp2, tmp5),
    vsubq_f32(tmp1, tmp6),
    vsubq_f32(tmp0, tmp7)
  } };

  /* Store 4x4 blocks to the intermediate workspace, ready for the second pass.
   * (VST4 transposes the blocks.  We need to operate on rows in the next
   * pass.)
   */
  vst4q_f32(workspace_1, rows_0123);
  vst4q_f32(workspace_2, rows_4567);
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
 * Again, the original C implementation of the float IDCT (jpeg_idct_float())
 * can be found in jidctflt.c.  Algorithmic changes made here are documented
 * inline.
 */

static INLINE void jsimd_idct_float_pass2_regular(float32_t * restrict workspace,
                                                  JSAMPARRAY restrict output_buf,
                                                  JDIMENSION output_col,
                                                  unsigned buf_offset)
{
  /* Even part */

  /* Apply signed->unsigned and prepare float->int conversion */
  float32x4_t z5 = vld1q_f32(workspace + 0 * DCTSIZE / 2);
  z5 = vaddq_n_f32(z5, CENTERJSAMPLE + 0.5f);

  float32x4_t ws4 = vld1q_f32(workspace + 4 * DCTSIZE / 2);
  float32x4_t tmp10 = vaddq_f32(z5, ws4);
  float32x4_t tmp11 = vsubq_f32(z5, ws4);

  float32x4_t ws2 = vld1q_f32(workspace + 2 * DCTSIZE / 2);
  float32x4_t ws6 = vld1q_f32(workspace + 6 * DCTSIZE / 2);
  float32x4_t tmp13 = vaddq_f32(ws2, ws6);
  float32x4_t tmp12 = vsubq_f32(vmulq_n_f32(vsubq_f32(ws2, ws6), 1.414213562f), tmp13);

  float32x4_t tmp0 = vaddq_f32(tmp10, tmp13);
  float32x4_t tmp3 = vsubq_f32(tmp10, tmp13);
  float32x4_t tmp1 = vaddq_f32(tmp11, tmp12);
  float32x4_t tmp2 = vsubq_f32(tmp11, tmp12);

  /* Odd part */

  float32x4_t ws5 = vld1q_f32(workspace + 5 * DCTSIZE / 2);
  float32x4_t ws3 = vld1q_f32(workspace + 3 * DCTSIZE / 2);
  float32x4_t z13 = vaddq_f32(ws5, ws3);
  float32x4_t neg_z10 = vsubq_f32(ws3, ws5);
  float32x4_t ws1 = vld1q_f32(workspace + 1 * DCTSIZE / 2);
  float32x4_t ws7 = vld1q_f32(workspace + 7 * DCTSIZE / 2);
  float32x4_t z11 = vaddq_f32(ws1, ws7);
  float32x4_t z12 = vsubq_f32(ws1, ws7);

  float32x4_t tmp7 = vaddq_f32(z11, z13);
  tmp11 = vmulq_n_f32(vsubq_f32(z11, z13), 1.414213562f);

  z5 = vmulq_n_f32(vsubq_f32(z12, neg_z10), 1.847759065f); /* 2*c2 */
  tmp10 = vfmaq_n_f32(z5, z12, -1.082392200f);
  tmp12 = vfmaq_n_f32(z5, neg_z10, +2.613125930f);

  float32x4_t tmp6 = vsubq_f32(tmp12, tmp7);
  float32x4_t tmp5 = vsubq_f32(tmp11, tmp6);
  float32x4_t tmp4 = vsubq_f32(tmp10, tmp5);

  /* Final output stage: float->int, narrow to 16-bit, and pack */
  /* Can reinterpret rather than use MOVNB (assuming valid data -
   * if invalid, will just get nonsense)
   */
  uint16x8_t cols_02_u16 = vmovnq_u32(vcvtq_u32_f32(vaddq_f32(tmp0, tmp7)),
                                      vcvtq_u32_f32(vaddq_f32(tmp2, tmp5)));
  uint16x8_t cols_13_u16 = vmovnq_u32(vcvtq_u32_f32(vaddq_f32(tmp1, tmp6)),
                                      vcvtq_u32_f32(vaddq_f32(tmp3, tmp4)));
  uint16x8_t cols_46_u16 = vmovnq_u32(vcvtq_u32_f32(vsubq_f32(tmp3, tmp4)),
                                      vcvtq_u32_f32(vsubq_f32(tmp1, tmp6)));
  uint16x8_t cols_57_u16 = vmovnq_u32(vcvtq_u32_f32(vsubq_f32(tmp2, tmp5)),
                                      vcvtq_u32_f32(vsubq_f32(tmp0, tmp7)));
  /* Narrow to 8-bit, clamping to range [0-255], and pack */
  uint8x16_t cols_0123_u8 = vqmovnq_u16(cols_02_u16, cols_13_u16);
  uint8x16_t cols_4567_u8 = vqmovnq_u16(cols_46_u16, cols_57_u16);
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

static INLINE void jsimd_idct_float_pass2_sparse(float32_t * restrict workspace,
                                                 JSAMPARRAY restrict output_buf,
                                                 JDIMENSION output_col,
                                                 unsigned buf_offset)
{
  /* Even part */

  /* Apply signed->unsigned and prepare float->int conversion */
  float32x4_t z5 = vld1q_f32(workspace + 0 * DCTSIZE / 2);
  z5 = vaddq_n_f32(z5, CENTERJSAMPLE + 0.5f);

  float32x4_t ws2 = vld1q_f32(workspace + 2 * DCTSIZE / 2);
  float32x4_t tmp13 = ws2;
  float32x4_t tmp12 = vmulq_n_f32(ws2, 0.414213562f);

  float32x4_t tmp0 = vaddq_f32(z5, tmp13);
  float32x4_t tmp3 = vsubq_f32(z5, tmp13);
  float32x4_t tmp1 = vaddq_f32(z5, tmp12);
  float32x4_t tmp2 = vsubq_f32(z5, tmp12);

  /* Odd part */

  float32x4_t ws3 = vld1q_f32(workspace + 3 * DCTSIZE / 2);
  float32x4_t ws1 = vld1q_f32(workspace + 1 * DCTSIZE / 2);

  float32x4_t tmp7 = vaddq_f32(ws1, ws3);
  float32x4_t tmp11 = vmulq_n_f32(vsubq_f32(ws1, ws3), 1.414213562f);

  z5 = vmulq_n_f32(vsubq_f32(ws1, ws3), 1.847759065f); /* 2*c2 */
  float32x4_t tmp10 = vfmaq_n_f32(z5, ws1, -1.082392200f);
  tmp12 = vfmaq_n_f32(z5, ws3, 2.613125930f);

  float32x4_t tmp6 = vsubq_f32(tmp12, tmp7);
  float32x4_t tmp5 = vsubq_f32(tmp11, tmp6);
  float32x4_t tmp4 = vsubq_f32(tmp10, tmp5);

  /* Final output stage: float->int, narrow to 16-bit, and pack */
  /* Can reinterpret rather than use MOVNB (assuming valid data -
   * if invalid, will just get nonsense)
   */
  uint16x8_t cols_02_u16 = vmovnq_u32(vcvtq_u32_f32(vaddq_f32(tmp0, tmp7)),
                                      vcvtq_u32_f32(vaddq_f32(tmp2, tmp5)));
  uint16x8_t cols_13_u16 = vmovnq_u32(vcvtq_u32_f32(vaddq_f32(tmp1, tmp6)),
                                      vcvtq_u32_f32(vaddq_f32(tmp3, tmp4)));
  uint16x8_t cols_46_u16 = vmovnq_u32(vcvtq_u32_f32(vsubq_f32(tmp3, tmp4)),
                                      vcvtq_u32_f32(vsubq_f32(tmp1, tmp6)));
  uint16x8_t cols_57_u16 = vmovnq_u32(vcvtq_u32_f32(vsubq_f32(tmp2, tmp5)),
                                      vcvtq_u32_f32(vsubq_f32(tmp0, tmp7)));
  /* Narrow to 8-bit, clamping to range [0-255], and pack */
  uint8x16_t cols_0123_u8 = vqmovnq_u16(cols_02_u16, cols_13_u16);
  uint8x16_t cols_4567_u8 = vqmovnq_u16(cols_46_u16, cols_57_u16);
  /* Reinterpret as 32-bit units for final transpose */
  uint32x4_t cols_0123_u32 = vreinterpretq_u32_u8(cols_0123_u8);
  uint32x4_t cols_4567_u32 = vreinterpretq_u32_u8(cols_4567_u8);

  /* Last transposition through scatter stores */
  uint32x4_t outptrs = vld1q_u32((uint32_t *) (output_buf + buf_offset));
  outptrs = vaddq_n_u32(outptrs, output_col);
  vstrwq_scatter_base_u32(outptrs, 0, cols_0123_u32);
  vstrwq_scatter_base_u32(outptrs, 4, cols_4567_u32);
}

#endif // __ARM_FEATURE_MVE & 2
