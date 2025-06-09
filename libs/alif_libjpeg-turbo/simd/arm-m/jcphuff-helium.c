/*
 * jcphuff-helium.c - prepare data for progressive Huffman encoding (Arm Helium))
 *
 * Copyright (C) 2020-2021, Arm Limited.  All Rights Reserved.
 * Copyright (C) 2022, Matthieu Darbois.  All Rights Reserved.
 * Copyright (C) 2022, D. R. Commander.  All Rights Reserved.
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

/* Narrowing load instructions don't exist, but we can simulate them. */
static INLINE int16x8_t vldrwq_s16(const int32_t *base)
{
  int16x8x2_t result = vld2q_s16((const int16_t *) base);
#ifdef __ARM_BIG_ENDIAN
  return result.val[1];
#else
  return result.val[0];
#endif
}

/* Data preparation for encode_mcu_AC_first().
 *
 * The equivalent scalar C function (encode_mcu_AC_first_prepare()) can be
 * found in jcphuff.c.
 */

void jsimd_encode_mcu_AC_first_prepare_helium
  (const JCOEF * restrict block,
   const int * restrict jpeg_natural_order_start, int Sl, int Al,
   UJCOEF * restrict values, size_t * restrict zerobits)
{
  UJCOEF * restrict values_ptr = values;
  UJCOEF * restrict diff_values_ptr = values + DCTSIZE2;

  /* Rows of coefficients to zero (since they haven't been processed) */
  int i, rows_to_zero = 8;

  for (i = 0; i < Sl / 16; i++) {
    uint16x8_t order = vreinterpretq_u16_s16(vldrwq_s16(jpeg_natural_order_start + 0));
    int16x8_t coefs1 = vldrhq_gather_shifted_offset_s16(block, order);
    order = vreinterpretq_u16_s16(vldrwq_s16(jpeg_natural_order_start + 8));
    int16x8_t coefs2 = vldrhq_gather_shifted_offset_s16(block, order);

    /* Isolate sign of coefficients. */
    uint16x8_t sign_coefs1 = vreinterpretq_u16_s16(vshrq_n_s16(coefs1, 15));
    uint16x8_t sign_coefs2 = vreinterpretq_u16_s16(vshrq_n_s16(coefs2, 15));
    /* Compute absolute value of coefficients and apply point transform Al. */
    uint16x8_t abs_coefs1 = vreinterpretq_u16_s16(vabsq_s16(coefs1));
    uint16x8_t abs_coefs2 = vreinterpretq_u16_s16(vabsq_s16(coefs2));
    abs_coefs1 = vshlq_r_u16(abs_coefs1, -Al);
    abs_coefs2 = vshlq_r_u16(abs_coefs2, -Al);

    /* Compute diff values. */
    uint16x8_t diff1 = veorq_u16(abs_coefs1, sign_coefs1);
    uint16x8_t diff2 = veorq_u16(abs_coefs2, sign_coefs2);

    /* Store transformed coefficients and diff values. */
    vst1q_u16(values_ptr, abs_coefs1);
    vst1q_u16(values_ptr + DCTSIZE, abs_coefs2);
    vst1q_u16(diff_values_ptr, diff1);
    vst1q_u16(diff_values_ptr + DCTSIZE, diff2);
    values_ptr += 16;
    diff_values_ptr += 16;
    jpeg_natural_order_start += 16;
    rows_to_zero -= 2;
  }

  /* Same operation but for remaining partial vector */
  int remaining_coefs = Sl % 16;
  if (remaining_coefs > 8) {
    uint16x8_t order = vreinterpretq_u16_s16(vldrwq_s16(jpeg_natural_order_start + 0));
    int16x8_t coefs1 = vldrhq_gather_shifted_offset_s16(block, order);
    order = vreinterpretq_u16_s16(vldrwq_s16(jpeg_natural_order_start + 8));
    int16x8_t coefs2 = vldrhq_gather_shifted_offset_z_s16(block, order,
                                                          vctp16q(remaining_coefs - 8));

    /* Isolate sign of coefficients. */
    uint16x8_t sign_coefs1 = vreinterpretq_u16_s16(vshrq_n_s16(coefs1, 15));
    uint16x8_t sign_coefs2 = vreinterpretq_u16_s16(vshrq_n_s16(coefs2, 15));
    /* Compute absolute value of coefficients and apply point transform Al. */
    uint16x8_t abs_coefs1 = vreinterpretq_u16_s16(vabsq_s16(coefs1));
    uint16x8_t abs_coefs2 = vreinterpretq_u16_s16(vabsq_s16(coefs2));
    abs_coefs1 = vshlq_r_u16(abs_coefs1, -Al);
    abs_coefs2 = vshlq_r_u16(abs_coefs2, -Al);

    /* Compute diff values. */
    uint16x8_t diff1 = veorq_u16(abs_coefs1, sign_coefs1);
    uint16x8_t diff2 = veorq_u16(abs_coefs2, sign_coefs2);

    /* Store transformed coefficients and diff values. */
    vst1q_u16(values_ptr, abs_coefs1);
    vst1q_u16(values_ptr + DCTSIZE, abs_coefs2);
    vst1q_u16(diff_values_ptr, diff1);
    vst1q_u16(diff_values_ptr + DCTSIZE, diff2);
    values_ptr += 16;
    diff_values_ptr += 16;
    rows_to_zero -= 2;

  } else if (remaining_coefs > 0) {
    uint16x8_t order = vreinterpretq_u16_s16(vldrwq_s16(jpeg_natural_order_start + 0));
    int16x8_t coefs = vldrhq_gather_shifted_offset_z_s16(block, order,
                                                         vctp16q(remaining_coefs));

    /* Isolate sign of coefficients. */
    uint16x8_t sign_coefs = vreinterpretq_u16_s16(vshrq_n_s16(coefs, 15));
    /* Compute absolute value of coefficients and apply point transform Al. */
    uint16x8_t abs_coefs = vreinterpretq_u16_s16(vabsq_s16(coefs));
    abs_coefs = vshlq_r_u16(abs_coefs, -Al);

    /* Compute diff values. */
    uint16x8_t diff = veorq_u16(abs_coefs, sign_coefs);

    /* Store transformed coefficients and diff values. */
    vst1q_u16(values_ptr, abs_coefs);
    vst1q_u16(diff_values_ptr, diff);
    values_ptr += 8;
    diff_values_ptr += 8;
    rows_to_zero--;
  }

  /* Zero remaining memory in the values and diff_values blocks. */
  for (i = 0; i < rows_to_zero; i++) {
    vst1q_u16(values_ptr, vdupq_n_u16(0));
    vst1q_u16(diff_values_ptr, vdupq_n_u16(0));
    values_ptr += 8;
    diff_values_ptr += 8;
  }

  /* Construct zerobits bitmap.  A set bit means that the corresponding
   * coefficient != 0.
   */
  uint16x8_t row0 = vld1q_u16(values + 0 * DCTSIZE);
  uint16x8_t row1 = vld1q_u16(values + 1 * DCTSIZE);
  uint16x8_t row2 = vld1q_u16(values + 2 * DCTSIZE);
  uint16x8_t row3 = vld1q_u16(values + 3 * DCTSIZE);
  uint16x8_t row4 = vld1q_u16(values + 4 * DCTSIZE);
  uint16x8_t row5 = vld1q_u16(values + 5 * DCTSIZE);
  uint16x8_t row6 = vld1q_u16(values + 6 * DCTSIZE);
  uint16x8_t row7 = vld1q_u16(values + 7 * DCTSIZE);

  const uint16x8_t bitmap_mask =
    vcreateq_u16(0x0008000400020001, 0x0080004000200010);

  /* Replace non-zero lanes with a 1-bit lane indicator */
  uint16x8_t row0_ne0 = vpselq_u16(bitmap_mask, row0, vcmpneq_n_u16(row0, 0));
  uint16x8_t row1_ne0 = vpselq_u16(bitmap_mask, row1, vcmpneq_n_u16(row1, 0));
  uint16x8_t row2_ne0 = vpselq_u16(bitmap_mask, row2, vcmpneq_n_u16(row2, 0));
  uint16x8_t row3_ne0 = vpselq_u16(bitmap_mask, row3, vcmpneq_n_u16(row3, 0));
  uint16x8_t row4_ne0 = vpselq_u16(bitmap_mask, row4, vcmpneq_n_u16(row4, 0));
  uint16x8_t row5_ne0 = vpselq_u16(bitmap_mask, row5, vcmpneq_n_u16(row5, 0));
  uint16x8_t row6_ne0 = vpselq_u16(bitmap_mask, row6, vcmpneq_n_u16(row6, 0));
  uint16x8_t row7_ne0 = vpselq_u16(bitmap_mask, row7, vcmpneq_n_u16(row7, 0));

  /* Sum all lanes to get an 8-bit bitmask, and combine 4 of these */
  uint32_t bitmap_rows_0123 = vaddvq_u16(row3_ne0);
  bitmap_rows_0123 = vaddvaq_u16(bitmap_rows_0123 << 8, row2_ne0);
  bitmap_rows_0123 = vaddvaq_u16(bitmap_rows_0123 << 8, row1_ne0);
  bitmap_rows_0123 = vaddvaq_u16(bitmap_rows_0123 << 8, row0_ne0);
  uint32_t bitmap_rows_4567 = vaddvq_u16(row7_ne0);
  bitmap_rows_4567 = vaddvaq_u16(bitmap_rows_4567 << 8, row6_ne0);
  bitmap_rows_4567 = vaddvaq_u16(bitmap_rows_4567 << 8, row5_ne0);
  bitmap_rows_4567 = vaddvaq_u16(bitmap_rows_4567 << 8, row4_ne0);
  /* Store zerobits bitmap. */
  zerobits[0] = bitmap_rows_0123;
  zerobits[1] = bitmap_rows_4567;
}


/* Data preparation for encode_mcu_AC_refine().
 *
 * The equivalent scalar C function (encode_mcu_AC_refine_prepare()) can be
 * found in jcphuff.c.
 */

int jsimd_encode_mcu_AC_refine_prepare_helium
  (const JCOEF * restrict block,
   const int * restrict jpeg_natural_order_start, int Sl, int Al,
   UJCOEF *restrict absvalues, size_t * restrict bits)
{
  /* Temporary storage buffers for data used to compute the signbits bitmap and
   * the end-of-block (EOB) position
   */
  uint8_t coef_sign_bits[64];
  mve_pred16_t coef_eq1_p[8];

  UJCOEF *absvalues_ptr = absvalues;
  uint8_t *coef_sign_bits_ptr = coef_sign_bits;
  mve_pred16_t *eq1_p_ptr = coef_eq1_p;

  /* Rows of coefficients to zero (since they haven't been processed) */
  int i, rows_to_zero = 8;

  for (i = 0; i < Sl / 16; i++) {
    uint16x8_t order = vreinterpretq_u16_s16(vldrwq_s16(jpeg_natural_order_start + 0));
    int16x8_t coefs1 = vldrhq_gather_shifted_offset_s16(block, order);
    order = vreinterpretq_u16_s16(vldrwq_s16(jpeg_natural_order_start + 8));
    int16x8_t coefs2 = vldrhq_gather_shifted_offset_s16(block, order);

    /* Compute and store data for signbits bitmap. */
    uint16x8_t sign_coefs1 = vreinterpretq_u16_s16(vshrq_n_s16(coefs1, 15));
    uint16x8_t sign_coefs2 = vreinterpretq_u16_s16(vshrq_n_s16(coefs2, 15));
    vstrbq_u16(coef_sign_bits_ptr, sign_coefs1);
    vstrbq_u16(coef_sign_bits_ptr + DCTSIZE, sign_coefs2);

    /* Compute absolute value of coefficients and apply point transform Al. */
    uint16x8_t abs_coefs1 = vreinterpretq_u16_s16(vabsq_s16(coefs1));
    uint16x8_t abs_coefs2 = vreinterpretq_u16_s16(vabsq_s16(coefs2));
    abs_coefs1 = vshlq_r_u16(abs_coefs1, -Al);
    abs_coefs2 = vshlq_r_u16(abs_coefs2, -Al);
    vst1q_u16(absvalues_ptr, abs_coefs1);
    vst1q_u16(absvalues_ptr + DCTSIZE, abs_coefs2);

    /* Test whether transformed coefficient values == 1 (used to find EOB
     * position.)
     */
    eq1_p_ptr[0] = vcmpeqq_n_u16(abs_coefs1, 1);
    eq1_p_ptr[1] = vcmpeqq_n_u16(abs_coefs2, 1);

    absvalues_ptr += 16;
    coef_sign_bits_ptr += 16;
    eq1_p_ptr += 2;
    jpeg_natural_order_start += 16;
    rows_to_zero -= 2;
  }

  /* Same operation but for remaining partial vector */
  int remaining_coefs = Sl % 16;
  if (remaining_coefs > 8) {
    uint16x8_t order = vreinterpretq_u16_s16(vldrwq_s16(jpeg_natural_order_start + 0));
    int16x8_t coefs1 = vldrhq_gather_shifted_offset_s16(block, order);
    order = vreinterpretq_u16_s16(vldrwq_s16(jpeg_natural_order_start + 8));
    int16x8_t coefs2 = vldrhq_gather_shifted_offset_z_s16(block, order,
                                                          vctp16q(remaining_coefs - 8));

    /* Compute and store data for signbits bitmap. */
    uint16x8_t sign_coefs1 = vreinterpretq_u16_s16(vshrq_n_s16(coefs1, 15));
    uint16x8_t sign_coefs2 = vreinterpretq_u16_s16(vshrq_n_s16(coefs2, 15));
    vstrbq_u16(coef_sign_bits_ptr, sign_coefs1);
    vstrbq_u16(coef_sign_bits_ptr + DCTSIZE, sign_coefs2);

    /* Compute absolute value of coefficients and apply point transform Al. */
    uint16x8_t abs_coefs1 = vreinterpretq_u16_s16(vabsq_s16(coefs1));
    uint16x8_t abs_coefs2 = vreinterpretq_u16_s16(vabsq_s16(coefs2));
    abs_coefs1 = vshlq_r_u16(abs_coefs1, -Al);
    abs_coefs2 = vshlq_r_u16(abs_coefs2, -Al);
    vst1q_u16(absvalues_ptr, abs_coefs1);
    vst1q_u16(absvalues_ptr + DCTSIZE, abs_coefs2);

    /* Test whether transformed coefficient values == 1 (used to find EOB
     * position.)
     */
    eq1_p_ptr[0] = vcmpeqq_n_u16(abs_coefs1, 1);
    eq1_p_ptr[1] = vcmpeqq_n_u16(abs_coefs2, 1);

    absvalues_ptr += 16;
    coef_sign_bits_ptr += 16;
    eq1_p_ptr += 2;
    jpeg_natural_order_start += 16;
    rows_to_zero -= 2;

  } else if (remaining_coefs > 0) {
    uint16x8_t order = vreinterpretq_u16_s16(vldrwq_s16(jpeg_natural_order_start + 0));
    int16x8_t coefs = vldrhq_gather_shifted_offset_z_s16(block, order,
                                                         vctp16q(remaining_coefs));


    /* Compute and store data for signbits bitmap. */
    uint16x8_t sign_coefs = vreinterpretq_u16_s16(vshrq_n_s16(coefs, 15));
    vstrbq_u16(coef_sign_bits_ptr, sign_coefs);

    /* Compute absolute value of coefficients and apply point transform Al. */
    uint16x8_t abs_coefs = vreinterpretq_u16_s16(vabsq_s16(coefs));
    abs_coefs = vshlq_r_u16(abs_coefs, -Al);
    vst1q_u16(absvalues_ptr, abs_coefs);

    /* Test whether transformed coefficient values == 1 (used to find EOB
     * position.)
     */
    eq1_p_ptr[0] = vcmpeqq_n_u16(abs_coefs, 1);

    absvalues_ptr += 8;
    coef_sign_bits_ptr += 8;
    eq1_p_ptr++;
    rows_to_zero--;
  }

  /* Zero remaining memory in blocks. */
  for (i = 0; i < rows_to_zero; i++) {
    vst1q_u16(absvalues_ptr, vdupq_n_u16(0));
    vstrbq_u16(coef_sign_bits_ptr, vdupq_n_u16(0));
    eq1_p_ptr[0] = 0;
    absvalues_ptr += 8;
    coef_sign_bits_ptr += 8;
    eq1_p_ptr++;
  }

  /* Construct zerobits bitmap. */
  /* It may seem like it would make more sense to store predicates as we
   * work through above, and examine them, but Arm Compiler seems quite bad
   * at manipulating predicates as data, being reluctant to just assign registers
   * normally (it will do VSTR p0->stack, VLDR stack->p0,
   * VMRS p0->reg, STR reg->array, rather than just VSTR p0->array).
   */
  uint16x8_t abs_row0 = vld1q_u16(absvalues + 0 * DCTSIZE);
  uint16x8_t abs_row1 = vld1q_u16(absvalues + 1 * DCTSIZE);
  uint16x8_t abs_row2 = vld1q_u16(absvalues + 2 * DCTSIZE);
  uint16x8_t abs_row3 = vld1q_u16(absvalues + 3 * DCTSIZE);
  uint16x8_t abs_row4 = vld1q_u16(absvalues + 4 * DCTSIZE);
  uint16x8_t abs_row5 = vld1q_u16(absvalues + 5 * DCTSIZE);
  uint16x8_t abs_row6 = vld1q_u16(absvalues + 6 * DCTSIZE);
  uint16x8_t abs_row7 = vld1q_u16(absvalues + 7 * DCTSIZE);

  const uint16x8_t bitmap_mask =
    vcreateq_u16(0x0008000400020001, 0x0080004000200010);

  /* Replace non-zero lanes with a 1-bit lane indicator */
  uint16x8_t abs_row0_ne0 = vpselq_u16(bitmap_mask, abs_row0, vcmpneq_n_u16(abs_row0, 0));
  uint16x8_t abs_row1_ne0 = vpselq_u16(bitmap_mask, abs_row1, vcmpneq_n_u16(abs_row1, 0));
  uint16x8_t abs_row2_ne0 = vpselq_u16(bitmap_mask, abs_row2, vcmpneq_n_u16(abs_row2, 0));
  uint16x8_t abs_row3_ne0 = vpselq_u16(bitmap_mask, abs_row3, vcmpneq_n_u16(abs_row3, 0));
  uint16x8_t abs_row4_ne0 = vpselq_u16(bitmap_mask, abs_row4, vcmpneq_n_u16(abs_row4, 0));
  uint16x8_t abs_row5_ne0 = vpselq_u16(bitmap_mask, abs_row5, vcmpneq_n_u16(abs_row5, 0));
  uint16x8_t abs_row6_ne0 = vpselq_u16(bitmap_mask, abs_row6, vcmpneq_n_u16(abs_row6, 0));
  uint16x8_t abs_row7_ne0 = vpselq_u16(bitmap_mask, abs_row7, vcmpneq_n_u16(abs_row7, 0));

  /* Sum all lanes to get an 8-bit bitmask, and combine 4 of these */
  uint32_t bitmap_rows_0123 = vaddvq_u16(abs_row3_ne0);
  bitmap_rows_0123 = vaddvaq_u16(bitmap_rows_0123 << 8, abs_row2_ne0);
  bitmap_rows_0123 = vaddvaq_u16(bitmap_rows_0123 << 8, abs_row1_ne0);
  bitmap_rows_0123 = vaddvaq_u16(bitmap_rows_0123 << 8, abs_row0_ne0);
  uint32_t bitmap_rows_4567 = vaddvq_u16(abs_row7_ne0);
  bitmap_rows_4567 = vaddvaq_u16(bitmap_rows_4567 << 8, abs_row6_ne0);
  bitmap_rows_4567 = vaddvaq_u16(bitmap_rows_4567 << 8, abs_row5_ne0);
  bitmap_rows_4567 = vaddvaq_u16(bitmap_rows_4567 << 8, abs_row4_ne0);
  /* Store zerobits bitmap. */
  bits[0] = bitmap_rows_0123;
  bits[1] = bitmap_rows_4567;

  /* Construct signbits bitmap. */
  uint16x8_t signbits_row0 = vldrbq_u16(coef_sign_bits + 0 * DCTSIZE);
  uint16x8_t signbits_row1 = vldrbq_u16(coef_sign_bits + 1 * DCTSIZE);
  uint16x8_t signbits_row2 = vldrbq_u16(coef_sign_bits + 2 * DCTSIZE);
  uint16x8_t signbits_row3 = vldrbq_u16(coef_sign_bits + 3 * DCTSIZE);
  uint16x8_t signbits_row4 = vldrbq_u16(coef_sign_bits + 4 * DCTSIZE);
  uint16x8_t signbits_row5 = vldrbq_u16(coef_sign_bits + 5 * DCTSIZE);
  uint16x8_t signbits_row6 = vldrbq_u16(coef_sign_bits + 6 * DCTSIZE);
  uint16x8_t signbits_row7 = vldrbq_u16(coef_sign_bits + 7 * DCTSIZE);

  signbits_row0 = vbicq_u16(bitmap_mask, signbits_row0);
  signbits_row1 = vbicq_u16(bitmap_mask, signbits_row1);
  signbits_row2 = vbicq_u16(bitmap_mask, signbits_row2);
  signbits_row3 = vbicq_u16(bitmap_mask, signbits_row3);
  signbits_row4 = vbicq_u16(bitmap_mask, signbits_row4);
  signbits_row5 = vbicq_u16(bitmap_mask, signbits_row5);
  signbits_row6 = vbicq_u16(bitmap_mask, signbits_row6);
  signbits_row7 = vbicq_u16(bitmap_mask, signbits_row7);

  bitmap_rows_0123 = vaddvq_u16(signbits_row3);
  bitmap_rows_0123 = vaddvaq_u16(bitmap_rows_0123 << 8, signbits_row2);
  bitmap_rows_0123 = vaddvaq_u16(bitmap_rows_0123 << 8, signbits_row1);
  bitmap_rows_0123 = vaddvaq_u16(bitmap_rows_0123 << 8, signbits_row0);
  bitmap_rows_4567 = vaddvq_u16(signbits_row7);
  bitmap_rows_4567 = vaddvaq_u16(bitmap_rows_4567 << 8, signbits_row6);
  bitmap_rows_4567 = vaddvaq_u16(bitmap_rows_4567 << 8, signbits_row5);
  bitmap_rows_4567 = vaddvaq_u16(bitmap_rows_4567 << 8, signbits_row4);

  /* Store signbits bitmap. */
  bits[2] = bitmap_rows_0123;
  bits[3] = bitmap_rows_4567;

  /* Construct bitmap to find EOB position (the index of the last coefficient
   * equal to 1.) We stored predicates with 2 bits per coefficient.
   * (Most of this "repacking" is eliminated by the compiler - it realises
   * it can just load from the array).
   */
  /* Combine to four 32-bit masks for 2 rows each */
  uint32_t bitmap_2_rows_01 = ((uint32_t) coef_eq1_p[1] << 16) | coef_eq1_p[0];
  uint32_t bitmap_2_rows_23 = ((uint32_t) coef_eq1_p[3] << 16) | coef_eq1_p[2];
  uint32_t bitmap_2_rows_45 = ((uint32_t) coef_eq1_p[5] << 16) | coef_eq1_p[4];
  uint32_t bitmap_2_rows_67 = ((uint32_t) coef_eq1_p[7] << 16) | coef_eq1_p[6];

  /* Combine to two 64-bit masks for 4 rows each */
  uint64_t bitmap_2_rows_0123 = ((uint64_t) bitmap_2_rows_23 << 32) | bitmap_2_rows_01;
  uint64_t bitmap_2_rows_4567 = ((uint64_t) bitmap_2_rows_67 << 32) | bitmap_2_rows_45;

  /* Return EOB position. */
  if (bitmap_2_rows_0123 == 0 && bitmap_2_rows_4567 == 0) {
    /* EOB position is defined to be 0 if all coefficients != 1. */
    return 0;
  } else if (bitmap_2_rows_4567 != 0) {
    return 63 - __builtin_clzll(bitmap_2_rows_4567) / 2;
  } else {
    return 31 - __builtin_clzll(bitmap_2_rows_0123) / 2;
  }
}
