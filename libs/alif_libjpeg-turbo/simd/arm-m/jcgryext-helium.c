/*
 * jcgryext-neon.c - grayscale colorspace conversion (Arm Neon)
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

/* This file is included by jcgray-neon.c */


/* RGB -> Grayscale conversion is defined by the following equation:
 *    Y  =  0.29900 * R + 0.58700 * G + 0.11400 * B
 *
 * Avoid floating point arithmetic by using shifted integer constants:
 *    0.29899597 = 19595 * 2^-16
 *    0.58700561 = 38470 * 2^-16
 *    0.11399841 =  7471 * 2^-16
 * These constants are defined in jcgray-neon.c
 *
 * This is the same computation as the RGB -> Y portion of RGB -> YCbCr.
 */

void jsimd_rgb_gray_convert_helium(JDIMENSION image_width, JSAMPARRAY restrict input_buf,
                                   JSAMPIMAGE restrict output_buf, JDIMENSION output_row,
                                   int num_rows)
{
  while (--num_rows >= 0) {
    JSAMPROW restrict inptr = *input_buf++;
    JSAMPROW restrict outptr = output_buf[0][output_row];
    output_row++;

    int cols_remaining = image_width;
    for (; cols_remaining >= 16; cols_remaining -= 16) {
#if RGB_PIXELSIZE == 4
      uint8x16x4_t input_pixels = vld4q_u8(inptr);
      uint16x8_t r_b = vmovlbq_u8(input_pixels.val[RGB_RED]);
      uint16x8_t g_b = vmovlbq_u8(input_pixels.val[RGB_GREEN]);
      uint16x8_t b_b = vmovlbq_u8(input_pixels.val[RGB_BLUE]);
#else
      const uint8x16_t offsets = vmulq_n_u8(vidupq_n_u8(0, 1), RGB_PIXELSIZE);
      uint16x8_t r_b = vldrbq_gather_offset_u16(inptr + RGB_RED, offsets);
      uint16x8_t g_b = vldrbq_gather_offset_u16(inptr + RGB_GREEN, offsets);
      uint16x8_t b_b = vldrbq_gather_offset_u16(inptr + RGB_BLUE, offsets);
#endif

      /* Compute Y = 0.29900 * R + 0.58700 * G + 0.11400 * B */
      uint32x4_t y_bb = vmullbq_int_u16(r_b, vdupq_n_u16(F_0_298));
      uint32x4_t y_bt = vmulltq_int_u16(r_b, vdupq_n_u16(F_0_298));
      y_bb = vmlaq_n_u32(y_bb, vmovlbq_u16(g_b), F_0_587);
      y_bt = vmlaq_n_u32(y_bt, vmovltq_u16(g_b), F_0_587);
      y_bb = vmlaq_n_u32(y_bb, vmovlbq_u16(b_b), F_0_113);
      y_bt = vmlaq_n_u32(y_bt, vmovltq_u16(b_b), F_0_113);

      /* Descale Y values (rounding right shift) and narrow to 16-bit. */
      uint16x8_t y_b = vrshrnq_n_u32(y_bb, y_bt, 16);

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
      uint32x4_t y_tt = vmulltq_int_u16(r_t, vdupq_n_u16(F_0_298));
      y_tb = vmlaq_n_u32(y_tb, vmovlbq_u16(g_t), F_0_587);
      y_tt = vmlaq_n_u32(y_tt, vmovltq_u16(g_t), F_0_587);
      y_tb = vmlaq_n_u32(y_tb, vmovlbq_u16(b_t), F_0_113);
      y_tt = vmlaq_n_u32(y_tt, vmovltq_u16(b_t), F_0_113);

      /* Descale Y values (rounding right shift) and narrow to 16-bit. */
      uint16x8_t y_t = vrshrnq_n_u32(y_tb, y_tt, 16);

      /* Narrow Y values to 8-bit and store to memory.  Buffer overwrite is
       * permitted up to the next multiple of ALIGN_SIZE bytes.
       */
      vst1q_u8(outptr, vmovntq_u16(vreinterpretq_u8_u16(y_b), y_t));

      /* Increment pointers. */
      inptr += (16 * RGB_PIXELSIZE);
      outptr += 16;
    }
    for (; cols_remaining > 0; cols_remaining -= 8) {
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
      uint32x4_t y_t = vmulltq_int_u16(r, vdupq_n_u16(F_0_298));
      y_b = vmlaq_n_u32(y_b, vmovlbq_u16(g), F_0_587);
      y_t = vmlaq_n_u32(y_t, vmovltq_u16(g), F_0_587);
      y_b = vmlaq_n_u32(y_b, vmovlbq_u16(b), F_0_113);
      y_t = vmlaq_n_u32(y_t, vmovltq_u16(b), F_0_113);

      /* Descale Y values (rounding right shift) and narrow to 16-bit. */
      uint16x8_t y = vrshrnq_n_u32(y_b, y_t, 16);

      /* Narrow Y values to 8-bit and store to memory.  Buffer overwrite is
       * permitted up to the next multiple of ALIGN_SIZE bytes.
       */
      vstrbq_u16(outptr, y);

      /* Increment pointers. */
      inptr += (8 * RGB_PIXELSIZE);
      outptr += 8;
    }
  }
}
