/*
 * jquantf-helium.c - sample data conversion and quantization (Arm Helium)
 *
 * Copyright (C) 2020-2021, Arm Limited.  All Rights Reserved.
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

#include <math.h>
#include <arm_acle.h>
#include <arm_mve.h>

#if __ARM_FEATURE_MVE & 2

/* After downsampling, the resulting sample values are in the range [0, 255],
 * but the Discrete Cosine Transform (DCT) operates on values centered around
 * 0.
 *
 * To prepare sample values for the DCT, load samples into a DCT workspace,
 * subtracting CENTERJSAMPLE (128).  The samples, now in the range [-128, 127],
 * are also widened from 8- to 16-bit.
 *
 * The equivalent scalar C function convsamp_float() can be found in jcdctmgr.c.
 */

void jsimd_convsamp_float_helium(JSAMPARRAY restrict sample_data, JDIMENSION start_col,
                                 FAST_FLOAT * restrict workspace)
{
  for (int i = 0; i < DCTSIZE; i++) {
    uint32x4_t samp_row_l = vldrbq_u32(sample_data[i] + start_col + 0);
    uint32x4_t samp_row_r = vldrbq_u32(sample_data[i] + start_col + 4);
    int32x4_t row_l = vsubq_n_s32(vreinterpretq_s32_u32(samp_row_l), CENTERJSAMPLE);
    int32x4_t row_r = vsubq_n_s32(vreinterpretq_s32_u32(samp_row_r), CENTERJSAMPLE);
    float32x4_t out_l = vcvtq_f32_s32(row_l);
    float32x4_t out_r = vcvtq_f32_s32(row_r);
    vst1q_f32(workspace + i * DCTSIZE + 0, out_l);
    vst1q_f32(workspace + i * DCTSIZE + 4, out_r);
  }
}


/* After the DCT, the resulting array of coefficient values needs to be divided
 * by an array of quantization values.
 *
 * To avoid a slow division operation, the DCT coefficients are multiplied by
 * the reciprocals of the quantization values.
 *
 * The equivalent scalar C function quantize_float() can be found in jcdctmgr.c.
 */

void jsimd_quantize_float_helium(JCOEFPTR restrict coef_block,
                                 FAST_FLOAT * restrict divisors,
                                 FAST_FLOAT * restrict workspace)
{
  JCOEFPTR output_ptr = coef_block;
  int i;

// If not limited, Arm Compiler 6.19 goes nuts, and tries
// to unroll so much it runs out of registers and spills a load
// to stack.
#pragma clang loop unroll_count(4)
  for (i = 0; i < DCTSIZE2; i += 4) {
    /* Apply the quantization and scaling factor */
    float32x4_t row = vmulq_f32(vld1q_f32(workspace + i), vld1q_f32(divisors + i));

    /* Round to nearest integer. */
    vstrhq_s32(output_ptr + i, vcvtnq_s32_f32(row));
  }
}

#endif // __ARM_FEATURE_MVE & 2

