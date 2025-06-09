/*
 * jdcolext-helium.c - colorspace conversion (Arm Helium)
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

/* This file is included by jdcolor-helium.c. */


/* YCbCr -> RGB conversion is defined by the following equations:
 *    R = Y                        + 1.40200 * (Cr - 128)
 *    G = Y - 0.34414 * (Cb - 128) - 0.71414 * (Cr - 128)
 *    B = Y + 1.77200 * (Cb - 128)
 *
 * Scaled integer constants are used to avoid floating-point arithmetic:
 *    0.3441467 = 11277 * 2^-15
 *    0.7141418 = 23401 * 2^-15
 *    0.4020081 = 13173 * 2^-15
 *    0.7720032 = 25297 * 2^-15
 * These constants are defined in jdcolor-helium.c.
 *
 * To ensure correct results, rounding is used when descaling.
 */

/* Notes on safe memory access for YCbCr -> RGB conversion routines:
 *
 * Input memory buffers can be safely overread up to the next multiple of
 * ALIGN_SIZE bytes, since they are always allocated by alloc_sarray() in
 * jmemmgr.c.
 *
 * The output buffer cannot safely be written beyond output_width, since
 * output_buf points to a possibly unpadded row in the decompressed image
 * buffer allocated by the calling program.
 */

//#define SLOPPY_GREEN
//#define SEMISLOPPY_GREEN

void jsimd_ycc_rgb_convert_helium(JDIMENSION output_width, JSAMPIMAGE restrict input_buf,
                                  JDIMENSION input_row, JSAMPARRAY restrict output_buf,
                                  int num_rows)
{
  JSAMPROW restrict outptr;
  /* Pointers to Y, Cb, and Cr data */
  JSAMPROW restrict inptr0, inptr1, inptr2;

  while (--num_rows >= 0) {
    inptr0 = input_buf[0][input_row];
    inptr1 = input_buf[1][input_row];
    inptr2 = input_buf[2][input_row];
    input_row++;
    outptr = *output_buf++;
    int cols_remaining = output_width;
    for (; cols_remaining >= 16; cols_remaining -= 16) {
#if 0
      uint16x8_t cb = vld1q_u8(inptr1);
      uint16x8_t cr = vld1q_u8(inptr2);
      int8x16_t cr_128 = vreinterpretq_s8_u8(vsubq_n_u8(cr, 128));
      int8x16_t cb_128 = vreinterpretq_s8_u8(vsubq_n_u8(cb, 128));
      int16x8_t y_sub_g_b = vaddq_s16(vqdmullbq_n_s8(cb_128, F8_0_344),
                                      vqdmullbq_n_s8(cr_128, F8_0_714));
      int16x8_t y_sub_g_t = vaddq_s16(vqdmulltq_n_s8(cb_128, F8_0_344),
                                      vqdmulltq_n_s8(cr_128, F8_0_714));
#else
      /* Do bottom and top sequentially, reloading, rather than
       * in parallel like the Neon, to reduce register pressure.
       */
      int16x8_t cb_b = vreinterpretq_s16_u16(vmovlbq_u8(vld1q_u8(inptr1)));
      int16x8_t cr_b = vreinterpretq_s16_u16(vmovlbq_u8(vld1q_u8(inptr2)));
      /* Subtract 128 from Cb and Cr. */
      int16x8_t cr_128_b = vsubq_n_s16(cr_b, 128);
      int16x8_t cb_128_b = vsubq_n_s16(cb_b, 128);
      /* Compute Y-G: 0.34414 * (Cb - 128) + 0.71414 * (Cr - 128) */
#ifdef SLOPPY_GREEN
      int16x8_t y_sub_g_b = vqrdmulhq_n_s16(cb_128_b, F_0_344);
      y_sub_g_b = vqrdmlahq_n_s16(y_sub_g_b, cr_128_b, F_0_714);
#elif defined SEMISLOPPY_GREEN
      int16x8_t y_sub_g_b = vqrdmulhq_n_s16(vshlq_n_s16(cb_128_b, 8), F_0_344);
      y_sub_g_b = vqdmlahq_n_s16(y_sub_g_b, vshlq_n_s16(cr_128_b, 8), F_0_714);
      y_sub_g_b = vrshrq_n_s16(y_sub_g_b, 8);
#else
     int32x4_t y_sub_g_bb = vaddq_s32(vqdmullbq_n_s16(cb_128_b, F_0_344),
                                       vqdmullbq_n_s16(cr_128_b, F_0_714));
      int32x4_t y_sub_g_bt = vaddq_s32(vqdmulltq_n_s16(cb_128_b, F_0_344),
                                       vqdmulltq_n_s16(cr_128_b, F_0_714));
      /* Descale G components: shift right 16, round, and narrow to 16-bit. */
      int16x8_t y_sub_g_b = vrshrnq_n_s32(y_sub_g_bb, y_sub_g_bt, 16);
#endif
      /* Compute R-Y: 1.40200 * (Cr - 128) */
      int16x8_t r_sub_y_b = vqrdmlahq_n_s16(cr_128_b, cr_128_b, F_0_402);
      /* Compute B-Y: 1.77200 * (Cb - 128) */
      int16x8_t b_sub_y_b = vqrdmlahq_n_s16(cb_128_b, cb_128_b, F_0_772);
      /* Add Y. */
      int16x8_t y_b  = vreinterpretq_s16_u16(vmovlbq_u8(vld1q_u8(inptr0)));
      int16x8_t r_b = vaddq_s16(y_b, r_sub_y_b);
      int16x8_t b_b = vaddq_s16(y_b, b_sub_y_b);
      int16x8_t g_b = vsubq_s16(y_b, y_sub_g_b);

      /* Repeat for the top pixels */
      int16x8_t cb_t = vreinterpretq_s16_u16(vmovltq_u8(vld1q_u8(inptr1)));
      int16x8_t cr_t = vreinterpretq_s16_u16(vmovltq_u8(vld1q_u8(inptr2)));
      /* Subtract 128 from Cb and Cr. */
      int16x8_t cr_128_t = vsubq_n_s16(cr_t, 128);
      int16x8_t cb_128_t = vsubq_n_s16(cb_t, 128);
      /* Compute Y-G: 0.34414 * (Cb - 128) + 0.71414 * (Cr - 128) */
#ifdef SLOPPY_GREEN
      int16x8_t y_sub_g_t = vqrdmulhq_n_s16(cb_128_t, F_0_344);
      y_sub_g_t = vqrdmlahq_n_s16(y_sub_g_t, cr_128_t, F_0_714);
#elif defined SEMISLOPPY_GREEN
      int16x8_t y_sub_g_t = vqrdmulhq_n_s16(vshlq_n_s16(cb_128_t, 8), F_0_344);
      y_sub_g_t = vqdmlahq_n_s16(y_sub_g_t, vshlq_n_s16(cr_128_t, 8), F_0_714);
      y_sub_g_t = vrshrq_n_s16(y_sub_g_t, 8);
#else
      int32x4_t y_sub_g_tb = vaddq_s32(vqdmullbq_n_s16(cb_128_t, F_0_344),
                                       vqdmullbq_n_s16(cr_128_t, F_0_714));
      int32x4_t y_sub_g_tt = vaddq_s32(vqdmulltq_n_s16(cb_128_t, F_0_344),
                                       vqdmulltq_n_s16(cr_128_t, F_0_714));
      /* Descale G components: shift right 16, round, and narrow to 16-bit. */
      int16x8_t y_sub_g_t = vrshrnq_n_s32(y_sub_g_tb, y_sub_g_tt, 16);
#endif
      /* Compute R-Y: 1.40200 * (Cr - 128) */
      int16x8_t r_sub_y_t = vqrdmlahq_n_s16(cr_128_t, cr_128_t, F_0_402);
      /* Compute B-Y: 1.77200 * (Cb - 128) */
      int16x8_t b_sub_y_t = vqrdmlahq_n_s16(cb_128_t, cb_128_t, F_0_772);
      /* Add Y. */
      int16x8_t y_t = vreinterpretq_s16_u16(vmovltq_u8(vld1q_u8(inptr0)));
      int16x8_t r_t = vaddq_s16(y_t, r_sub_y_t);
      int16x8_t b_t = vaddq_s16(y_t, b_sub_y_t);
      int16x8_t g_t = vsubq_s16(y_t, y_sub_g_t);
#endif

#if RGB_PIXELSIZE == 4
      uint8x16x4_t rgba;
      /* Convert each component to unsigned and narrow, clamping to [0-255]. */
      rgba.val[RGB_RED]   = vqmovunq_s16(r_b, r_t);
      rgba.val[RGB_GREEN] = vqmovunq_s16(g_b, g_t);
      rgba.val[RGB_BLUE]  = vqmovunq_s16(b_b, b_t);
      /* Set alpha channel to opaque (0xFF). */
      rgba.val[RGB_ALPHA] = vdupq_n_u8(0xFF);
      /* Store RGBA pixel data to memory. */
      vst4q_u8(outptr, rgba);
#elif RGB_PIXELSIZE == 3
      /* Convert each component to unsigned and narrow, clamping to [0-255]. */
      uint8x16_t r = vqmovunq_s16(r_b, r_t);
      uint8x16_t g = vqmovunq_s16(g_b, g_t);
      uint8x16_t b = vqmovunq_s16(b_b, b_t);
      /* Store RGB pixel data to memory. */
      const uint8x16_t offsets = vmulq_n_u8(vidupq_n_u8(0, 1), RGB_PIXELSIZE);
      vstrbq_scatter_offset_u8(outptr + RGB_RED, offsets, r);
      vstrbq_scatter_offset_u8(outptr + RGB_GREEN, offsets, g);
      vstrbq_scatter_offset_u8(outptr + RGB_BLUE, offsets, b);
#else
      /* Pack R, G, and B values in ratio 5:6:5. */
      uint16x8x2_t rgb565;
      rgb565.val[0] = vqshluq_n_s16(r_b, 8);
      rgb565.val[0] = vsriq_n_u16(rgb565.val[0], vqshluq_n_s16(g_b, 8), 5);
      rgb565.val[0] = vsriq_n_u16(rgb565.val[0], vqshluq_n_s16(b_b, 8), 11);
      rgb565.val[1] = vqshluq_n_s16(r_t, 8);
      rgb565.val[1] = vsriq_n_u16(rgb565.val[1], vqshluq_n_s16(g_t, 8), 5);
      rgb565.val[1] = vsriq_n_u16(rgb565.val[1], vqshluq_n_s16(b_t, 8), 11);
      /* Store RGB pixel data to memory. */
      vst2q_u16((uint16_t *)outptr, rgb565);
#endif

      /* Increment pointers. */
      inptr0 += 16;
      inptr1 += 16;
      inptr2 += 16;
      outptr += (RGB_PIXELSIZE * 16);
    }
    for (; cols_remaining > 0; cols_remaining -= 8) {
      /* This can't be a TP loop, because of the wider ops in the green path,
       * so we we only predicate the loads and stores to reduce predication overhead.
       */
      mve_pred16_t p = vctp16q(cols_remaining);
      int16x8_t cb = vreinterpretq_s16_u16(vldrbq_z_u16(inptr1, p));
      int16x8_t cr = vreinterpretq_s16_u16(vldrbq_z_u16(inptr2, p));
      /* Subtract 128 from Cb and Cr. */
      int16x8_t cr_128 = vsubq_n_u16(cr, 128);
      int16x8_t cb_128 = vsubq_n_u16(cb, 128);
      /* Compute Y-G: 0.34414 * (Cb - 128) + 0.71414 * (Cr - 128) */
#ifdef SLOPPY_GREEN
      int16x8_t y_sub_g = vqrdmulhq_n_s16(cb_128, F_0_344);
      y_sub_g = vqrdmlahq_n_s16(y_sub_g, cr_128, F_0_714);
#elif defined SEMISLOPPY_GREEN
      int16x8_t y_sub_g = vqrdmulhq_n_s16(vshlq_n_s16(cb_128, 8), F_0_344);
      y_sub_g = vqdmlahq_n_s16(y_sub_g, vshlq_n_s16(cr_128, 8), F_0_714);
      y_sub_g = vrshrq_n_s16(y_sub_g, 8);
#else
      int32x4_t y_sub_g_b = vaddq_s32(vqdmullbq_n_s16(cb_128, F_0_344),
                                      vqdmullbq_n_s16(cr_128, F_0_714));
      int32x4_t y_sub_g_t = vaddq_s32(vqdmulltq_n_s16(cb_128, F_0_344),
                                      vqdmulltq_n_s16(cr_128, F_0_714));
      /* Descale G components: shift right 16, round, and narrow to 16-bit. */
      int16x8_t y_sub_g = vrshrnq_n_s32(y_sub_g_b, y_sub_g_t, 16);
#endif
      /* Compute R-Y: 1.40200 * (Cr - 128) */
      int16x8_t r_sub_y = vqrdmlahq_n_s16(cr_128, cr_128, F_0_402);
      /* Compute B-Y: 1.77200 * (Cb - 128) */
      int16x8_t b_sub_y = vqrdmlahq_n_s16(cb_128, cb_128, F_0_772);
      /* Add Y. */
      int16x8_t y = vreinterpretq_s16_u16(vldrbq_z_u16(inptr0, p));
      int16x8_t r = vaddq_s16(y, r_sub_y);
      int16x8_t b = vaddq_s16(y, b_sub_y);
      int16x8_t g = vsubq_s16(y, y_sub_g);

#if RGB_PIXELSIZE == 4
      uint16x8x4_t rgba;
      /* Convert each component to unsigned and narrow, clamping to [0-255]. */
      rgba.val[RGB_RED] = vreinterpretq_u16_u8(vqmovunbq_s16(vuninitializedq_u8(), r));
      rgba.val[RGB_GREEN] = vreinterpretq_u16_u8(vqmovunbq_s16(vuninitializedq_u8(), g));
      rgba.val[RGB_BLUE] = vreinterpretq_u16_u8(vqmovunbq_s16(vuninitializedq_u8(), b));
      /* Set alpha channel to opaque (0xFF). */
      rgba.val[RGB_ALPHA] = vdupq_n_u16(0xFF);
      /* Store RGBA pixel data to memory. */
      const uint16x8_t offsets = vidupq_n_u16(0, RGB_PIXELSIZE);
      vstrbq_scatter_offset_p_u16(outptr + RGB_RED, offsets, rgba.val[RGB_RED], p);
      vstrbq_scatter_offset_p_u16(outptr + RGB_GREEN, offsets, rgba.val[RGB_GREEN], p);
      vstrbq_scatter_offset_p_u16(outptr + RGB_BLUE, offsets, rgba.val[RGB_BLUE], p);
      vstrbq_scatter_offset_p_u16(outptr + RGB_ALPHA, offsets, rgba.val[RGB_ALPHA], p);
#elif RGB_PIXELSIZE == 3
      /* Convert each component to unsigned and narrow, clamping to [0-255]. */
      r = vreinterpretq_u16_u8(vqmovunbq_s16(vuninitializedq_u8(), r));
      g = vreinterpretq_u16_u8(vqmovunbq_s16(vuninitializedq_u8(), g));
      b = vreinterpretq_u16_u8(vqmovunbq_s16(vuninitializedq_u8(), b));
      /* Store RGB pixel data to memory. */
      const uint16x8_t offsets = vmulq_n_u16(vidupq_n_u16(0, 1), RGB_PIXELSIZE);
      vstrbq_scatter_offset_p_u16(outptr + RGB_RED, offsets, r, p);
      vstrbq_scatter_offset_p_u16(outptr + RGB_GREEN, offsets, g, p);
      vstrbq_scatter_offset_p_u16(outptr + RGB_BLUE, offsets, b, p);
#else
      /* Pack R, G, and B values in ratio 5:6:5. */
      uint16x8_t rgb565 = vqshluq_n_s16(r, 8);
      rgb565 = vsriq_n_u16(rgb565, vqshluq_n_s16(g, 8), 5);
      rgb565 = vsriq_n_u16(rgb565, vqshluq_n_s16(b, 8), 11);
      /* Store RGB pixel data to memory. */
      vst1q_p_u16((uint16_t *)outptr, rgb565, p);
#endif

      /* Increment pointers. */
      inptr0 += 8;
      inptr1 += 8;
      inptr2 += 8;
      outptr += (RGB_PIXELSIZE * 8);
    }
  }
}
