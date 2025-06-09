/*
 * jchuff-helium.c - Huffman entropy encoding (Arm Helium)
 * Copyright (C) 2023, Alif Semiconductor.  All Rights Reserved.
 *
 * Copyright (C) 2020-2021, Arm Limited.  All Rights Reserved.
 * Copyright (C) 2020, 2022, D. R. Commander.  All Rights Reserved.
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
 *
 * NOTE: All referenced figures are from
 * Recommendation ITU-T T.81 (1992) | ISO/IEC 10918-1:1994.
 */

#define JPEG_INTERNALS
#include "../../jinclude.h"
#include "../../jpeglib.h"
#include "../../jsimd.h"
#include "../../jdct.h"
#include "../../jsimddct.h"
#include "../jsimd.h"
#include "../arm/jchuff.h"

#include <limits.h>

#include <arm_acle.h>
#include <arm_mve.h>

static INLINE
void process_two_rows(JCOEFPTR restrict block,
                      const uint8_t * restrict order_table,
                      uint8_t * restrict block_nbits,
                      uint16_t * restrict block_diff,
                      int row, int last_dc_val)
{
  /* Load rows of coefficients from DCT block in zig-zag order. */
  int16x8_t row0 = vldrhq_gather_shifted_offset_s16(
                       block, vldrbq_u16(order_table + (row + 0) * DCTSIZE));
  int16x8_t row1 = vldrhq_gather_shifted_offset_s16(
                       block, vldrbq_u16(order_table + (row + 1) * DCTSIZE));

  /* Compute DC coefficient difference value (F.1.1.5.1). */
  if (row == 0) { /* Should be inlined and hence compile-time */
    row0 = vsetq_lane_s16(block[0] - last_dc_val, row0, 0);
  }

  uint16x8_t abs_row0 = vreinterpretq_u16_s16(vabsq_s16(row0));
  uint16x8_t abs_row1 = vreinterpretq_u16_s16(vabsq_s16(row1));

  uint16x8_t row0_lz = vclzq_u16(abs_row0);
  uint16x8_t row1_lz = vclzq_u16(abs_row1);

  /* Compute nbits needed to specify magnitude of each coefficient. */
  //int16x8_t row0_minus_lz =  vnegq_s16(vreinterpretq_u16_s16(row0_lz));
  //int16x8_t row1_minus_lz =  vnegq_s16(vreinterpretq_u16_s16(row1_lz));

  uint16x8_t row0_nbits = vsubq_u16(vdupq_n_u16(16), row0_lz);
  uint16x8_t row1_nbits = vsubq_u16(vdupq_n_u16(16), row1_lz);
  //uint16x8_t row0_nbits = vaddq_n_u16(row0_minus_lz, 16);
  //uint16x8_t row1_nbits = vaddq_n_u16(row1_minus_lz, 16);

  vstrbq_u16(block_nbits + (row + 0) * DCTSIZE, row0_nbits);
  vstrbq_u16(block_nbits + (row + 1) * DCTSIZE, row1_nbits);

#if 1 // 1,486,389 (std dct)
  uint16x8_t row0_mask =
    vshlq_u16(vreinterpretq_u16_s16(vshrq_n_s16(row0, 15)),
              vnegq_s16(vreinterpretq_u16_s16(row0_lz)));
  uint16x8_t row1_mask =
    vshlq_u16(vreinterpretq_u16_s16(vshrq_n_s16(row1, 15)),
              vnegq_s16(vreinterpretq_u16_s16(row1_lz)));

  uint16x8_t row0_diff = veorq_u16(abs_row0, row0_mask);
  uint16x8_t row1_diff = veorq_u16(abs_row1, row1_mask);
#elif 1 // 1,487,289 - compiler does SUB R,R anyway
  uint16x8_t row0_mask =
    vshlq_u16(vreinterpretq_u16_s16(vshrq_n_s16(row0, 15)),
              row0_minus_lz);
  uint16x8_t row1_mask =
    vshlq_u16(vreinterpretq_u16_s16(vshrq_n_s16(row1, 15)),
              row1_minus_lz);

  uint16x8_t row0_diff = veorq_u16(abs_row0, row0_mask);
  uint16x8_t row1_diff = veorq_u16(abs_row1, row1_mask);
#elif 0 // 1,496,289
  uint16x8_t row0_mask =
    vshlq_u16(vdupq_n_u16(0xFFFF),
              vnegq_s16(vreinterpretq_u16_s16(row0_lz)));
  mve_pred16_t row0_lt0 = vcmpltq_n_s16(row0, 0);
  uint16x8_t row0_diff = veorq_m_u16(abs_row0, abs_row0, row0_mask, row0_lt0);
  uint16x8_t row1_mask =
    vshlq_u16(vdupq_n_u16(0xFFFF),
              vnegq_s16(vreinterpretq_u16_s16(row1_lz)));
  mve_pred16_t row1_lt0 = vcmpltq_n_s16(row1, 0);
  uint16x8_t row1_diff = veorq_m_u16(abs_row1, abs_row1, row1_mask, row1_lt0);
#elif 0 // 1,523,729
  uint16x8_t row0_mask =
    vshlq_u16(vdupq_n_u16(0xFFFF),
              vreinterpretq_s16_u16(row0_nbits));
  uint16x8_t row1_mask =
    vshlq_u16(vdupq_n_u16(0xFFFF),
              vreinterpretq_s16_u16(row1_nbits));
  mve_pred16_t row0_lt0 = vcmpltq_n_s16(row0, 0);
  mve_pred16_t row1_lt0 = vcmpltq_n_s16(row1, 0);
  uint16x8_t row0_diff = vmvnq_m_u16(abs_row0, abs_row0, row0_lt0);
  uint16x8_t row1_diff = vmvnq_m_u16(abs_row1, abs_row1, row1_lt0);
  row0_diff = vandq_u16(row0_diff, row0_mask);
  row1_diff = vandq_u16(row1_diff, row1_mask);
#else // 1,534,210
  uint16x8_t row0_mask =
    vshlq_u16(vdupq_n_u16(0xFFFF),
              vreinterpretq_s16_u16(row0_nbits));
  uint16x8_t row1_mask =
    vshlq_u16(vdupq_n_u16(0xFFFF),
              vreinterpretq_s16_u16(row1_nbits));
  mve_pred16_t row0_lt0 = vcmpltq_n_s16(row0, 0);
  mve_pred16_t row1_lt0 = vcmpltq_n_s16(row1, 0);
  uint16x8_t row0_diff = vbicq_m_u16(abs_row0, row0_mask, abs_row0, row0_lt0);
  uint16x8_t row1_diff = vbicq_m_u16(abs_row1, row1_mask, abs_row1, row1_lt0);
  row0_diff = vandq_m_u16(row0_diff, row0_mask, abs_row0, vpnot(row0_lt0));
  row1_diff = vandq_m_u16(row0_diff, row1_mask, abs_row1, vpnot(row1_lt0));
#endif

  /* Store diff values for rows 0 and 1. */
  vst1q_u16(block_diff + (row + 0) * DCTSIZE, row0_diff);
  vst1q_u16(block_diff + (row + 1) * DCTSIZE, row1_diff);
}

JOCTET *jsimd_huff_encode_one_block_helium(void * restrict state, JOCTET * restrict buffer,
                                           JCOEFPTR restrict block, int last_dc_val,
                                           c_derived_tbl * restrict dctbl,
                                           c_derived_tbl * restrict actbl,
                                           const UINT8 *order_table)
{
  uint8_t block_nbits[DCTSIZE2];
  uint16_t block_diff[DCTSIZE2];

  process_two_rows(block, order_table, block_nbits, block_diff, 0, last_dc_val);
  process_two_rows(block, order_table, block_nbits, block_diff, 2, 0);
  process_two_rows(block, order_table, block_nbits, block_diff, 4, 0);
  process_two_rows(block, order_table, block_nbits, block_diff, 6, 0);

  /* Construct bitmap to accelerate encoding of AC coefficients.  A set bit
   * means that the corresponding coefficient != 0.
   */
  mve_pred16_t row10_nbits_ne0 = vcmpneq_n_u8(vld1q_u8(block_nbits + 0 * DCTSIZE), 0);
  mve_pred16_t row32_nbits_ne0 = vcmpneq_n_u8(vld1q_u8(block_nbits + 2 * DCTSIZE), 0);
  mve_pred16_t row54_nbits_ne0 = vcmpneq_n_u8(vld1q_u8(block_nbits + 4 * DCTSIZE), 0);
  mve_pred16_t row76_nbits_ne0 = vcmpneq_n_u8(vld1q_u8(block_nbits + 6 * DCTSIZE), 0);

  uint32_t bitmap_0_31 = __rbit((row32_nbits_ne0 << 16) | (row10_nbits_ne0 << 0));
  uint32_t bitmap_32_63 = __rbit((row76_nbits_ne0 << 16) | (row54_nbits_ne0 << 0));

  /* Set up state and bit buffer for output bitstream. */
  working_state *state_ptr = state;
  int free_bits = state_ptr->cur.free_bits;
  size_t put_buffer = state_ptr->cur.put_buffer;

  /* Encode DC coefficient. */

  unsigned int nbits = block_nbits[0];
  /* Emit Huffman-coded symbol and additional diff bits. */
  unsigned int diff = block_diff[0];
  PUT_CODE(dctbl->ehufco[nbits], dctbl->ehufsi[nbits], diff)
  bitmap_0_31 <<= 1;

  /* Encode AC coefficients. */

  unsigned int r = 0;  /* r = run length of zeros */
  unsigned int i = 1;  /* i = number of coefficients encoded */

  while (bitmap_0_31 != 0) {
    r = __builtin_clz(bitmap_0_31);
    i += r;
    bitmap_0_31 <<= r;
    nbits = block_nbits[i];
    diff = block_diff[i];
    while (UNLIKELY(r > 15)) {
      /* If run length > 15, emit special run-length-16 codes. */
      /* (Holding the codes in local variables impairs overall performance) */
      PUT_BITS(actbl->ehufco[0xf0], actbl->ehufsi[0xf0]);
      r -= 16;
    }
    /* Emit Huffman symbol for run length / number of bits. (F.1.2.2.1) */
    unsigned int rs = (r << 4) + nbits;
    PUT_CODE(actbl->ehufco[rs], actbl->ehufsi[rs], diff)
    i++;
    bitmap_0_31 <<= 1;
  }

  r = 32 - i;
  i = 32;

  while (UNLIKELY(bitmap_32_63 != 0)) {
    unsigned int leading_zeros = __builtin_clz(bitmap_32_63);
    r += leading_zeros;
    i += leading_zeros;
    bitmap_32_63 <<= leading_zeros;
    nbits = block_nbits[i];
    diff = block_diff[i];
    while (UNLIKELY(r > 15)) {
      /* If run length > 15, emit special run-length-16 codes. */
      PUT_BITS(actbl->ehufco[0xf0], actbl->ehufsi[0xf0]);
      r -= 16;
    }
    /* Emit Huffman symbol for run length / number of bits. (F.1.2.2.1) */
    unsigned int rs = (r << 4) + nbits;
    PUT_CODE(actbl->ehufco[rs], actbl->ehufsi[rs], diff)
    r = 0;
    i++;
    bitmap_32_63 <<= 1;
  }

  /* If the last coefficient(s) were zero, emit an end-of-block (EOB) code.
   * The value of RS for the EOB code is 0.
   */
  if (LIKELY(i != 64)) {
    PUT_BITS(actbl->ehufco[0], actbl->ehufsi[0])
  }

  state_ptr->cur.put_buffer = put_buffer;
  state_ptr->cur.free_bits = free_bits;

  return buffer;
}
