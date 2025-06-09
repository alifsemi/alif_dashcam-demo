/*
 * jfdctflt-helium.c - floating-point FDCT (Arm Helium)
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

#if __ARM_FEATURE_MVE & 2

/* Forward declaration of FDCT helper functions */

static INLINE void jsimd_fdct_float_pass1(float32_t *data);

static INLINE void jsimd_fdct_float_pass2(float32_t *data);


/* Perform forward DCT on one block of coefficients.  For
 * reference, the C implementation (jpeg_fdct_float()) can be found in
 * jfdctflt.c.
 */

void jsimd_fdct_float_helium(FAST_FLOAT *data)
{
  /* Compute IDCT first pass on top 8x4 coefficient block. */

  jsimd_fdct_float_pass1(data);

  /* Compute IDCT first pass on bottom 8x4 coefficient block. */

  jsimd_fdct_float_pass1(data + 4 * DCTSIZE);

  /* Second pass: compute FDCT on columns. */

  jsimd_fdct_float_pass2(data);
  jsimd_fdct_float_pass2(data + 4);
}


/* Perform the first pass of the floating-point forward DCT on a
 * 8x4 block of coefficients.  (To process the full 8x8 DCT block, this
 * function needs to be called for both the top and bottom 8x4 blocks,
 *
 * The original C implementation of the float FDCT (jpeg_fdct_float()) can be
 * found in jfdctflt.c.
 */

static INLINE void jsimd_fdct_float_pass1(float32_t *data)
{
  const uint32x4_t offsets = vidupq_n_u32(0, DCTSIZE);

  float32x4_t col0 = vldrwq_gather_shifted_offset_f32(data + 0, offsets);
  float32x4_t col7 = vldrwq_gather_shifted_offset_f32(data + 7, offsets);
  float32x4_t tmp0 = vaddq_f32(col0, col7);
  float32x4_t tmp7 = vsubq_f32(col0, col7);
  float32x4_t col1 = vldrwq_gather_shifted_offset_f32(data + 1, offsets);
  float32x4_t col6 = vldrwq_gather_shifted_offset_f32(data + 6, offsets);
  float32x4_t tmp1 = vaddq_f32(col1, col6);
  float32x4_t tmp6 = vsubq_f32(col1, col6);
  float32x4_t col2 = vldrwq_gather_shifted_offset_f32(data + 2, offsets);
  float32x4_t col5 = vldrwq_gather_shifted_offset_f32(data + 5, offsets);
  float32x4_t tmp2 = vaddq_f32(col2, col5);
  float32x4_t tmp5 = vsubq_f32(col2, col5);
  float32x4_t col3 = vldrwq_gather_shifted_offset_f32(data + 3, offsets);
  float32x4_t col4 = vldrwq_gather_shifted_offset_f32(data + 4, offsets);
  float32x4_t tmp3 = vaddq_f32(col3, col4);
  float32x4_t tmp4 = vsubq_f32(col3, col4);

  /* Even part */

  float32x4_t tmp10 = vaddq_f32(tmp0, tmp3);
  float32x4_t tmp13 = vsubq_f32(tmp0, tmp3);
  float32x4_t tmp11 = vaddq_f32(tmp1, tmp2);
  float32x4_t tmp12 = vsubq_f32(tmp1, tmp2);

  col0 = vaddq_f32(tmp10, tmp11);
  col4 = vsubq_f32(tmp10, tmp11);

  float32x4_t z1 = vmulq_n_f32(vaddq_f32(tmp12, tmp13), 0.707106781f);
  col2 = vaddq_f32(tmp13, z1);
  col6 = vsubq_f32(tmp13, z1);

  /* Odd part */

  tmp10 = vaddq_f32(tmp4, tmp5);
  tmp11 = vaddq_f32(tmp5, tmp6);
  tmp12 = vaddq_f32(tmp6, tmp7);

  float32x4_t z5 = vmulq_n_f32(vsubq_f32(tmp10, tmp12), 0.382683433f);
  float32x4_t z2 = vfmaq_n_f32(z5, tmp10, 0.541196100f);
  float32x4_t z4 = vfmaq_n_f32(z5, tmp12, 1.306562965f);
  float32x4_t z3 = vmulq_n_f32(tmp11, 0.707106781f);

  float32x4_t z11 = vaddq_f32(tmp7, z3);
  float32x4_t z13 = vsubq_f32(tmp7, z3);

  col5 = vaddq_f32(z13, z2);
  col3 = vsubq_f32(z13, z2);
  col1 = vaddq_f32(z11, z4);
  col7 = vsubq_f32(z11, z4);

  vstrwq_scatter_shifted_offset_f32(data + 0, offsets, col0);
  vstrwq_scatter_shifted_offset_f32(data + 1, offsets, col1);
  vstrwq_scatter_shifted_offset_f32(data + 2, offsets, col2);
  vstrwq_scatter_shifted_offset_f32(data + 3, offsets, col3);
  vstrwq_scatter_shifted_offset_f32(data + 4, offsets, col4);
  vstrwq_scatter_shifted_offset_f32(data + 5, offsets, col5);
  vstrwq_scatter_shifted_offset_f32(data + 6, offsets, col6);
  vstrwq_scatter_shifted_offset_f32(data + 7, offsets, col7);
}


/* Perform the second pass of the accurate inverse DCT on a 4x8 block of
 * coefficients.  (To process the full 8x8 DCT block, this function-- or some
 * other optimized variant-- needs to be called for both the right and left 4x8
 * blocks.)
 *
 * Again, the original C implementation of the floating FDCT (jpeg_fdct_float())
 * can be found in jfdctflt.c.
 */

static INLINE void jsimd_fdct_float_pass2(float32_t *data)
{
  float32x4_t row0 = vld1q_f32(data + DCTSIZE * 0);
  float32x4_t row7 = vld1q_f32(data + DCTSIZE * 7);
  float32x4_t tmp0 = vaddq_f32(row0, row7);
  float32x4_t tmp7 = vsubq_f32(row0, row7);
  float32x4_t row1 = vld1q_f32(data + DCTSIZE * 1);
  float32x4_t row6 = vld1q_f32(data + DCTSIZE * 6);
  float32x4_t tmp1 = vaddq_f32(row1, row6);
  float32x4_t tmp6 = vsubq_f32(row1, row6);
  float32x4_t row2 = vld1q_f32(data + DCTSIZE * 2);
  float32x4_t row5 = vld1q_f32(data + DCTSIZE * 5);
  float32x4_t tmp2 = vaddq_f32(row2, row5);
  float32x4_t tmp5 = vsubq_f32(row2, row5);
  float32x4_t row3 = vld1q_f32(data + DCTSIZE * 3);
  float32x4_t row4 = vld1q_f32(data + DCTSIZE * 4);
  float32x4_t tmp3 = vaddq_f32(row3, row4);
  float32x4_t tmp4 = vsubq_f32(row3, row4);

  /* Even part */

  float32x4_t tmp10 = vaddq_f32(tmp0, tmp3);
  float32x4_t tmp13 = vsubq_f32(tmp0, tmp3);
  float32x4_t tmp11 = vaddq_f32(tmp1, tmp2);
  float32x4_t tmp12 = vsubq_f32(tmp1, tmp2);

  row0 = vaddq_f32(tmp10, tmp11);
  row4 = vsubq_f32(tmp10, tmp11);

  float32x4_t z1 = vmulq_n_f32(vaddq_f32(tmp12, tmp13), 0.707106781f);
  row2 = vaddq_f32(tmp13, z1);
  row6 = vsubq_f32(tmp13, z1);

  /* Odd part */

  tmp10 = vaddq_f32(tmp4, tmp5);
  tmp11 = vaddq_f32(tmp5, tmp6);
  tmp12 = vaddq_f32(tmp6, tmp7);

  float32x4_t z5 = vmulq_n_f32(vsubq_f32(tmp10, tmp12), 0.382683433f);
  float32x4_t z2 = vfmaq_n_f32(z5, tmp10, 0.541196100f);
  float32x4_t z4 = vfmaq_n_f32(z5, tmp12, 1.306562965f);
  float32x4_t z3 = vmulq_n_f32(tmp11, 0.707106781f);

  float32x4_t z11 = vaddq_f32(tmp7, z3);
  float32x4_t z13 = vsubq_f32(tmp7, z3);

  row5 = vaddq_f32(z13, z2);
  row3 = vsubq_f32(z13, z2);
  row1 = vaddq_f32(z11, z4);
  row7 = vsubq_f32(z11, z4);

  vst1q_f32(data + DCTSIZE * 0, row0);
  vst1q_f32(data + DCTSIZE * 1, row1);
  vst1q_f32(data + DCTSIZE * 2, row2);
  vst1q_f32(data + DCTSIZE * 3, row3);
  vst1q_f32(data + DCTSIZE * 4, row4);
  vst1q_f32(data + DCTSIZE * 5, row5);
  vst1q_f32(data + DCTSIZE * 6, row6);
  vst1q_f32(data + DCTSIZE * 7, row7);
}

#endif // __ARM_FEATURE_MVE & 2

