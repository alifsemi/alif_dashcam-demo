/*
 * jccolext-helium.c - colorspace conversion (Arm Helium)
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

/* This file is included by jccolor-helium.c */


/* RGB -> YCbCr conversion is defined by the following equations:
 *    Y  =  0.29900 * R + 0.58700 * G + 0.11400 * B
 *    Cb = -0.16874 * R - 0.33126 * G + 0.50000 * B  + 128
 *    Cr =  0.50000 * R - 0.41869 * G - 0.08131 * B  + 128
 *
 * Avoid floating point arithmetic by using shifted integer constants:
 *    0.29899597 = 19595 * 2^-16
 *    0.58700561 = 38470 * 2^-16
 *    0.11399841 =  7471 * 2^-16
 *    0.16874695 = 11059 * 2^-16
 *    0.33125305 = 21709 * 2^-16
 *    0.50000000 = 32768 * 2^-16
 *    0.41868592 = 27439 * 2^-16
 *    0.08131409 =  5329 * 2^-16
 * These constants are defined in jccolor-helium.c
 *
 * We add the fixed-point equivalent of 0.5 to Cb and Cr, which effectively
 * rounds up or down the result via integer truncation.
 */

void jsimd_rgb_ycc_convert_helium(JDIMENSION image_width, JSAMPARRAY restrict input_buf,
                                  JSAMPIMAGE restrict output_buf, JDIMENSION output_row,
                                  int num_rows)
{
  /* Pointer to RGB(X/A) input data */
  JSAMPROW restrict inptr;
  /* Pointers to Y, Cb, and Cr output data */
  JSAMPROW restrict outptr0, outptr1, outptr2;

  const uint32_t scaled_128_5 = (128 << 16) + 32767;

  while (--num_rows >= 0) {
    inptr = *input_buf++;
    outptr0 = output_buf[0][output_row];
    outptr1 = output_buf[1][output_row];
    outptr2 = output_buf[2][output_row];
    output_row++;

    int cols_remaining = image_width;
    for (; cols_remaining >= 16; cols_remaining -= 16) {
#if RGB_PIXELSIZE == 4
      uint8x16x4_t input_pixels = vld4q_u8(inptr);
      uint16x8_t r_b = vmovlbq_u8(input_pixels.val[RGB_RED]);
      uint16x8_t g_b = vmovlbq_u8(input_pixels.val[RGB_GREEN]);
      uint16x8_t b_b = vmovlbq_u8(input_pixels.val[RGB_BLUE]);
#else
      const uint16x8_t offsets = vmulq_n_u16(vidupq_n_u16(0, 2), RGB_PIXELSIZE);
      uint16x8_t r_b = vldrbq_gather_offset_u16(inptr + RGB_RED, offsets);
      uint16x8_t g_b = vldrbq_gather_offset_u16(inptr + RGB_GREEN, offsets);
      uint16x8_t b_b = vldrbq_gather_offset_u16(inptr + RGB_BLUE, offsets);
#endif
      /* Do bottom and top sequentially, rather than in parallel
       * like the Neon, to reduce register pressure.
       */

      /* Compute Y = 0.29900 * R + 0.58700 * G + 0.11400 * B */
      uint32x4_t y_bb = vmullbq_int_u16(r_b, vdupq_n_u16(F_0_298));
      y_bb = vmlaq_n_u32(y_bb, vmovlbq_u16(g_b), F_0_587);
      y_bb = vmlaq_n_u32(y_bb, vmovlbq_u16(b_b), F_0_113);
      uint32x4_t y_bt = vmulltq_int_u16(r_b, vdupq_n_u16(F_0_298));
      y_bt = vmlaq_n_u32(y_bt, vmovltq_u16(g_b), F_0_587);
      y_bt = vmlaq_n_u32(y_bt, vmovltq_u16(b_b), F_0_113);

      /* Compute Cb = -0.16874 * R - 0.33126 * G + 0.50000 * B  + 128 */
      uint32x4_t cb_bb = vdupq_n_u32(scaled_128_5);
      cb_bb = vmlaq_n_u32(cb_bb, vmovlbq_u16(r_b), -F_0_168);
      cb_bb = vmlaq_n_u32(cb_bb, vmovlbq_u16(g_b), -F_0_331);
      cb_bb = vmlaq_n_u32(cb_bb, vmovlbq_u16(b_b), F_0_500);
      uint32x4_t cb_bt = vdupq_n_u32(scaled_128_5);
      cb_bt = vmlaq_n_u32(cb_bt, vmovltq_u16(r_b), -F_0_168);
      cb_bt = vmlaq_n_u32(cb_bt, vmovltq_u16(g_b), -F_0_331);
      cb_bt = vmlaq_n_u32(cb_bt, vmovltq_u16(b_b), F_0_500);

      /* Compute Cr = 0.50000 * R - 0.41869 * G - 0.08131 * B  + 128 */
      uint32x4_t cr_bb = vdupq_n_u32(scaled_128_5);
      cr_bb = vmlaq_n_u32(cr_bb, vmovlbq_u16(r_b), F_0_500);
      cr_bb = vmlaq_n_u32(cr_bb, vmovlbq_u16(g_b), -F_0_418);
      cr_bb = vmlaq_n_u32(cr_bb, vmovlbq_u16(b_b), -F_0_081);
      uint32x4_t cr_bt = vdupq_n_u32(scaled_128_5);
      cr_bt = vmlaq_n_u32(cr_bt, vmovltq_u16(r_b), F_0_500);
      cr_bt = vmlaq_n_u32(cr_bt, vmovltq_u16(g_b), -F_0_418);
      cr_bt = vmlaq_n_u32(cr_bt, vmovltq_u16(b_b), -F_0_081);

      /* Descale Y values (rounding right shift) and narrow to 16-bit. */
      uint16x8_t y_b = vrshrnq_n_u32(y_bb, y_bt, 16);
      /* Descale Cb values (right shift) and narrow to 16-bit. */
      uint16x8_t cb_b = vshrnq_n_u32(cb_bb, cb_bt, 16);
      /* Descale Cr values (right shift) and narrow to 16-bit. */
      uint16x8_t cr_b = vshrnq_n_u32(cr_bb, cr_bt, 16);

      /* Repeat for the top pixels */
#if RGB_PIXELSIZE == 4
      uint16x8_t r_t = vmovltq_u8(input_pixels.val[RGB_RED]);
      uint16x8_t g_t = vmovltq_u8(input_pixels.val[RGB_GREEN]);
      uint16x8_t b_t = vmovltq_u8(input_pixels.val[RGB_BLUE]);
#else
      uint16x8_t r_t = vldrbq_gather_offset_u16(inptr + RGB_RED + RGB_PIXELSIZE, offsets);
      uint16x8_t g_t = vldrbq_gather_offset_u16(inptr + RGB_GREEN + RGB_PIXELSIZE, offsets);
      uint16x8_t b_t = vldrbq_gather_offset_u16(inptr + RGB_BLUE + RGB_PIXELSIZE, offsets);
#endif

      /* Compute Y = 0.29900 * R + 0.58700 * G + 0.11400 * B */
      uint32x4_t y_tb = vmullbq_int_u16(r_t, vdupq_n_u16(F_0_298));
      y_tb = vmlaq_n_u32(y_tb, vmovlbq_u16(g_t), F_0_587);
      y_tb = vmlaq_n_u32(y_tb, vmovlbq_u16(b_t), F_0_113);
      uint32x4_t y_tt = vmulltq_int_u16(r_t, vdupq_n_u16(F_0_298));
      y_tt = vmlaq_n_u32(y_tt, vmovltq_u16(g_t), F_0_587);
      y_tt = vmlaq_n_u32(y_tt, vmovltq_u16(b_t), F_0_113);

      /* Compute Cb = -0.16874 * R - 0.33126 * G + 0.50000 * B  + 128 */
      uint32x4_t cb_tb = vdupq_n_u32(scaled_128_5);
      cb_tb = vmlaq_n_u32(cb_tb, vmovlbq_u16(r_t), -F_0_168);
      cb_tb = vmlaq_n_u32(cb_tb, vmovlbq_u16(g_t), -F_0_331);
      cb_tb = vmlaq_n_u32(cb_tb, vmovlbq_u16(b_t), F_0_500);
      uint32x4_t cb_tt = vdupq_n_u32(scaled_128_5);
      cb_tt = vmlaq_n_u32(cb_tt, vmovltq_u16(r_t), -F_0_168);
      cb_tt = vmlaq_n_u32(cb_tt, vmovltq_u16(g_t), -F_0_331);
      cb_tt = vmlaq_n_u32(cb_tt, vmovltq_u16(b_t), F_0_500);

      /* Compute Cr = 0.50000 * R - 0.41869 * G - 0.08131 * B  + 128 */
      uint32x4_t cr_tb = vdupq_n_u32(scaled_128_5);
      cr_tb = vmlaq_n_u32(cr_tb, vmovlbq_u16(r_t), F_0_500);
      cr_tb = vmlaq_n_u32(cr_tb, vmovlbq_u16(g_t), -F_0_418);
      cr_tb = vmlaq_n_u32(cr_tb, vmovlbq_u16(b_t), -F_0_081);
      uint32x4_t cr_tt = vdupq_n_u32(scaled_128_5);
      cr_tt = vmlaq_n_u32(cr_tt, vmovltq_u16(r_t), F_0_500);
      cr_tt = vmlaq_n_u32(cr_tt, vmovltq_u16(g_t), -F_0_418);
      cr_tt = vmlaq_n_u32(cr_tt, vmovltq_u16(b_t), -F_0_081);

      /* Descale Y values (rounding right shift) and narrow to 16-bit. */
      uint16x8_t y_t = vrshrnq_n_u32(y_tb, y_tt, 16);
      /* Descale Cb values (right shift) and narrow to 16-bit. */
      uint16x8_t cb_t = vshrnq_n_u32(cb_tb, cb_tt, 16);
      /* Descale Cr values (right shift) and narrow to 16-bit. */
      uint16x8_t cr_t = vshrnq_n_u32(cr_tb, cr_tt, 16);

      /* Narrow Y, Cb, and Cr values to 8-bit and store to memory.  Buffer
       * overwrite is permitted up to the next multiple of ALIGN_SIZE bytes.
       */
      uint8x16_t y = vmovnq_u16(y_b, y_t);
      uint8x16_t cb = vmovnq_u16(cb_b, cb_t);
      uint8x16_t cr = vmovnq_u16(cr_b, cr_t);
      vst1q_u8(outptr0, y);
      vst1q_u8(outptr1, cb);
      vst1q_u8(outptr2, cr);

      /* Increment pointers. */
      inptr += (16 * RGB_PIXELSIZE);
      outptr0 += 16;
      outptr1 += 16;
      outptr2 += 16;
    }
    for (; cols_remaining > 0; cols_remaining -= 8) {
      /* This can't be a TP loop, because of the wider ops in the green path,
       * so we we only predicate the loads to reduce predication overhead.
       */
      mve_pred16_t p = vctp16q(cols_remaining);
#if RGB_PIXELSIZE == 4
      const uint16x8_t offsets = vidupq_n_u16(0, RGB_PIXELSIZE);
#else
      const uint16x8_t offsets = vmulq_n_u16(vidupq_n_u16(0, 1), RGB_PIXELSIZE);
#endif
      uint16x8_t r = vldrbq_gather_offset_z_u16(inptr + RGB_RED, offsets, p);
      uint16x8_t g = vldrbq_gather_offset_z_u16(inptr + RGB_GREEN, offsets, p);
      uint16x8_t b = vldrbq_gather_offset_z_u16(inptr + RGB_BLUE, offsets, p);

      /* Compute Y = 0.29900 * R + 0.58700 * G + 0.11400 * B */
      uint32x4_t y_b = vmullbq_int_u16(r, vdupq_n_u16(F_0_298));
      y_b = vmlaq_n_u32(y_b, vmovlbq_u16(g), F_0_587);
      y_b = vmlaq_n_u32(y_b, vmovlbq_u16(b), F_0_113);
      uint32x4_t y_t = vmulltq_int_u16(r, vdupq_n_u16(F_0_298));
      y_t = vmlaq_n_u32(y_t, vmovltq_u16(g), F_0_587);
      y_t = vmlaq_n_u32(y_t, vmovltq_u16(b), F_0_113);

      /* Compute Cb = -0.16874 * R - 0.33126 * G + 0.50000 * B  + 128 */
      uint32x4_t cb_b = vdupq_n_u32(scaled_128_5);
      cb_b = vmlaq_n_u32(cb_b, vmovlbq_u16(r), -F_0_168);
      cb_b = vmlaq_n_u32(cb_b, vmovlbq_u16(g), -F_0_331);
      cb_b = vmlaq_n_u32(cb_b, vmovlbq_u16(b), F_0_500);
      uint32x4_t cb_t = vdupq_n_u32(scaled_128_5);
      cb_t = vmlaq_n_u32(cb_t, vmovltq_u16(r), -F_0_168);
      cb_t = vmlaq_n_u32(cb_t, vmovltq_u16(g), -F_0_331);
      cb_t = vmlaq_n_u32(cb_t, vmovltq_u16(b), F_0_500);

      /* Compute Cr = 0.50000 * R - 0.41869 * G - 0.08131 * B  + 128 */
      uint32x4_t cr_b = vdupq_n_u32(scaled_128_5);
      cr_b = vmlaq_n_u32(cr_b, vmovlbq_u16(r), F_0_500);
      cr_b = vmlaq_n_u32(cr_b, vmovlbq_u16(g), -F_0_418);
      cr_b = vmlaq_n_u32(cr_b, vmovlbq_u16(b), -F_0_081);
      uint32x4_t cr_t = vdupq_n_u32(scaled_128_5);
      cr_t = vmlaq_n_u32(cr_t, vmovltq_u16(r), F_0_500);
      cr_t = vmlaq_n_u32(cr_t, vmovltq_u16(g), -F_0_418);
      cr_t = vmlaq_n_u32(cr_t, vmovltq_u16(b), -F_0_081);

      /* Descale Y values (rounding right shift) and narrow to 16-bit. */
      uint16x8_t y = vrshrnq_n_u32(y_b, y_t, 16);
      /* Descale Cb values (right shift) and narrow to 16-bit. */
      uint16x8_t cb = vshrnq_n_u32(cb_b, cb_t, 16);
      /* Descale Cr values (right shift) and narrow to 16-bit. */
      uint16x8_t cr = vshrnq_n_u32(cr_b, cr_t, 16);

      /* Narrow Y, Cb, and Cr values to 8-bit and store to memory.  Buffer
       * overwrite is permitted up to the next multiple of ALIGN_SIZE bytes.
       */
      vstrbq_u16(outptr0, y);
      vstrbq_u16(outptr1, cb);
      vstrbq_u16(outptr2, cr);

      /* Increment pointers. */
      inptr += (8 * RGB_PIXELSIZE);
      outptr0 += 8;
      outptr1 += 8;
      outptr2 += 8;
    }
  }
}
