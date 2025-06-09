/*
 * jquanti-helium.c - sample data conversion and quantization (Arm Helium)
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

#include <arm_mve.h>


/* After downsampling, the resulting sample values are in the range [0, 255],
 * but the Discrete Cosine Transform (DCT) operates on values centered around
 * 0.
 *
 * To prepare sample values for the DCT, load samples into a DCT workspace,
 * subtracting CENTERJSAMPLE (128).  The samples, now in the range [-128, 127],
 * are also widened from 8- to 16-bit.
 *
 * The equivalent scalar C function convsamp() can be found in jcdctmgr.c.
 */

void jsimd_convsamp_helium(JSAMPARRAY restrict sample_data, JDIMENSION start_col,
                           DCTELEM * restrict workspace)
{
  uint16x8_t samp_row0 = vldrbq_u16(sample_data[0] + start_col);
  uint16x8_t samp_row1 = vldrbq_u16(sample_data[1] + start_col);
  uint16x8_t samp_row2 = vldrbq_u16(sample_data[2] + start_col);
  uint16x8_t samp_row3 = vldrbq_u16(sample_data[3] + start_col);
  uint16x8_t samp_row4 = vldrbq_u16(sample_data[4] + start_col);
  uint16x8_t samp_row5 = vldrbq_u16(sample_data[5] + start_col);
  uint16x8_t samp_row6 = vldrbq_u16(sample_data[6] + start_col);
  uint16x8_t samp_row7 = vldrbq_u16(sample_data[7] + start_col);

  int16x8_t row0 = vsubq_n_s16(vreinterpretq_s16_u16(samp_row0), CENTERJSAMPLE);
  int16x8_t row1 = vsubq_n_s16(vreinterpretq_s16_u16(samp_row1), CENTERJSAMPLE);
  int16x8_t row2 = vsubq_n_s16(vreinterpretq_s16_u16(samp_row2), CENTERJSAMPLE);
  int16x8_t row3 = vsubq_n_s16(vreinterpretq_s16_u16(samp_row3), CENTERJSAMPLE);
  int16x8_t row4 = vsubq_n_s16(vreinterpretq_s16_u16(samp_row4), CENTERJSAMPLE);
  int16x8_t row5 = vsubq_n_s16(vreinterpretq_s16_u16(samp_row5), CENTERJSAMPLE);
  int16x8_t row6 = vsubq_n_s16(vreinterpretq_s16_u16(samp_row6), CENTERJSAMPLE);
  int16x8_t row7 = vsubq_n_s16(vreinterpretq_s16_u16(samp_row7), CENTERJSAMPLE);

  vst1q_s16(workspace + 0 * DCTSIZE, row0);
  vst1q_s16(workspace + 1 * DCTSIZE, row1);
  vst1q_s16(workspace + 2 * DCTSIZE, row2);
  vst1q_s16(workspace + 3 * DCTSIZE, row3);
  vst1q_s16(workspace + 4 * DCTSIZE, row4);
  vst1q_s16(workspace + 5 * DCTSIZE, row5);
  vst1q_s16(workspace + 6 * DCTSIZE, row6);
  vst1q_s16(workspace + 7 * DCTSIZE, row7);
}


/* After the DCT, the resulting array of coefficient values needs to be divided
 * by an array of quantization values.
 *
 * To avoid a slow division operation, the DCT coefficients are multiplied by
 * the (scaled) reciprocals of the quantization values and then right-shifted.
 *
 * The equivalent scalar C function quantize() can be found in jcdctmgr.c.
 */

void jsimd_quantize_helium(JCOEFPTR restrict coef_block, DCTELEM * restrict divisors,
                           DCTELEM * restrict workspace)
{
  JCOEFPTR out_ptr = coef_block;
  UDCTELEM *recip_ptr = (UDCTELEM *)divisors;
  UDCTELEM *corr_ptr = (UDCTELEM *)divisors + DCTSIZE2;
  DCTELEM *shift_ptr = divisors + 3 * DCTSIZE2;
  int i;

  for (i = 0; i < DCTSIZE; i ++) {
    /* Load DCT coefficients. */
    int16x8_t row = vld1q_s16(workspace + i * DCTSIZE);
    /* Load reciprocals of quantization values. */
    uint16x8_t recip = vld1q_u16(recip_ptr + i * DCTSIZE);
    uint16x8_t corr = vld1q_u16(corr_ptr + i * DCTSIZE);
    int16x8_t shift = vld1q_s16(shift_ptr + i * DCTSIZE);

    /* Extract sign from coefficients. */
    mve_pred16_t sign_row = vcmpltq_n_s16(row, 0);
    /* Get absolute value of DCT coefficients. */
    uint16x8_t abs_row = vreinterpretq_u16_s16(vabsq_s16(row));
    /* Add correction. */
    abs_row = vaddq_u16(abs_row, corr);

    /* Multiply DCT coefficients by quantization reciprocals. */
    abs_row = vmulhq_u16(abs_row, recip);

    /* Since VSHR only supports an immediate as its second argument, negate the
     * shift value and shift left.
     */
    row = vreinterpretq_s16_u16(vshlq_u16(abs_row, vnegq_s16(shift)));

    /* Restore sign to original product. */
    row = vnegq_m_s16(row, row, sign_row);

    /* Store quantized coefficients to memory. */
    vst1q_s16(out_ptr + i * DCTSIZE, row);
  }
}
