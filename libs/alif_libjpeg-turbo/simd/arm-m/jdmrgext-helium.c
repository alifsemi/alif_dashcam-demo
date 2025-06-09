/*
 * jdmrgext-helium.c - merged upsampling/color conversion (Arm Helium)
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

/* This file is included by jdmerge-helium.c. */


/* These routines combine simple (non-fancy, i.e. non-smooth) h2v1 or h2v2
 * chroma upsampling and YCbCr -> RGB color conversion into a single function.
 *
 * As with the standalone functions, YCbCr -> RGB conversion is defined by the
 * following equations:
 *    R = Y                        + 1.40200 * (Cr - 128)
 *    G = Y - 0.34414 * (Cb - 128) - 0.71414 * (Cr - 128)
 *    B = Y + 1.77200 * (Cb - 128)
 *
 * Scaled integer constants are used to avoid floating-point arithmetic:
 *    0.3441467 = 11277 * 2^-15
 *    0.7141418 = 23401 * 2^-15
 *    0.4020081 = 13173 * 2^-15
 *    0.7720032 = 25297 * 2^-15
 * These constants are defined in jdmerge-helium.c.
 *
 * To ensure correct results, rounding is used when descaling.
 */

/* Notes on safe memory access for merged upsampling/YCbCr -> RGB conversion
 * routines:
 *
 * Input memory buffers can be safely overread up to the next multiple of
 * ALIGN_SIZE bytes, since they are always allocated by alloc_sarray() in
 * jmemmgr.c.
 *
 * The output buffer cannot safely be written beyond output_width, since
 * output_buf points to a possibly unpadded row in the decompressed image
 * buffer allocated by the calling program.
 */

/* Upsample and color convert for the case of 2:1 horizontal and 1:1 vertical.
 */

void jsimd_h2v1_merged_upsample_helium(JDIMENSION output_width,
                                       JSAMPIMAGE restrict input_buf,
                                       JDIMENSION in_row_group_ctr,
                                       JSAMPARRAY restrict output_buf)
{
  JSAMPROW restrict outptr;
  /* Pointers to Y, Cb, and Cr data */
  JSAMPROW restrict inptr0, inptr1, inptr2;

  inptr0 = input_buf[0][in_row_group_ctr];
  inptr1 = input_buf[1][in_row_group_ctr];
  inptr2 = input_buf[2][in_row_group_ctr];
  outptr = output_buf[0];

  int cols_remaining = output_width;
  for (; cols_remaining >= 16; cols_remaining -= 16) {
    int16x8_t cb = vreinterpretq_s16_u16(vldrbq_u16(inptr1));
    int16x8_t cr = vreinterpretq_s16_u16(vldrbq_u16(inptr2));
    /* Subtract 128 from Cb and Cr. */
    int16x8_t cr_128 = vsubq_n_s16(cr, 128);
    int16x8_t cb_128 = vsubq_n_s16(cb, 128);
    /* Compute Y-G: 0.34414 * (Cb - 128) + 0.71414 * (Cr - 128) */
    int32x4_t y_sub_g_b = vaddq_s32(vqdmullbq_n_s16(cb_128, F_0_344),
                                    vqdmullbq_n_s16(cr_128, F_0_714));
    int32x4_t y_sub_g_t = vaddq_s32(vqdmulltq_n_s16(cb_128, F_0_344),
                                    vqdmulltq_n_s16(cr_128, F_0_714));
    /* Descale G components: shift right 16, round, and narrow to 16-bit. */
    int16x8_t y_sub_g = vrshrnq_n_s32(y_sub_g_b, y_sub_g_t, 16);
    /* Compute R-Y: 1.40200 * (Cr - 128) */
    int16x8_t r_sub_y = vqrdmlahq_n_s16(cr_128, cr_128, F_0_402);
    /* Compute B-Y: 1.77200 * (Cb - 128) */
    int16x8_t b_sub_y = vqrdmlahq_n_s16(cb_128, cb_128, F_0_772);
    /* De-interleave Y component values into two separate vectors, one
     * containing the component values with even-numbered indices and one
     * containing the component values with odd-numbered indices.
     */
    uint8x16_t y = vld1q_u8(inptr0);
    int16x8_t y_even = vreinterpretq_s16_u16(vmovlbq_u8(y));
    int16x8_t y_odd = vreinterpretq_s16_u16(vmovltq_u8(y));
    /* Add the chroma-derived values (Y-G, R-Y, and B-Y) to both the "even" and
     * "odd" Y component values.  This effectively upsamples the chroma
     * components horizontally.
     */
    int16x8_t g_even = vsubq_s16(y_even, y_sub_g);
    int16x8_t r_even = vaddq_s16(y_even, r_sub_y);
    int16x8_t b_even = vaddq_s16(y_even, b_sub_y);
    int16x8_t g_odd  = vsubq_s16(y_odd,  y_sub_g);
    int16x8_t r_odd  = vaddq_s16(y_odd,  r_sub_y);
    int16x8_t b_odd  = vaddq_s16(y_odd,  b_sub_y);
    /* Convert each component to unsigned and narrow, clamping to [0-255].
     * Re-interleave the "even" and "odd" component values.
     */
    uint8x16_t r = vqmovunq_s16(r_even, r_odd);
    uint8x16_t g = vqmovunq_s16(g_even, g_odd);
    uint8x16_t b = vqmovunq_s16(b_even, b_odd);

#ifdef RGB_ALPHA
    uint8x16x4_t rgba;
    rgba.val[RGB_RED] = r;
    rgba.val[RGB_GREEN] = g;
    rgba.val[RGB_BLUE] = b;
    /* Set alpha channel to opaque (0xFF). */
    rgba.val[RGB_ALPHA] = vdupq_n_u8(0xFF);
    /* Store RGBA pixel data to memory. */
    vst4q_u8(outptr, rgba);
#else
    /* Store RGB pixel data to memory. */
    const uint8x16_t offsets = vmulq_n_u8(vidupq_n_u8(0, 1), RGB_PIXELSIZE);
    vstrbq_scatter_offset_u8(outptr + RGB_RED, offsets, r);
    vstrbq_scatter_offset_u8(outptr + RGB_GREEN, offsets, g);
    vstrbq_scatter_offset_u8(outptr + RGB_BLUE, offsets, b);
#endif

    /* Increment pointers. */
    inptr0 += 16;
    inptr1 += 8;
    inptr2 += 8;
    outptr += (RGB_PIXELSIZE * 16);
  }

  if (cols_remaining > 0) {
    int16x8_t cb = vreinterpretq_s16_u16(vldrbq_u16(inptr1));
    int16x8_t cr = vreinterpretq_s16_u16(vldrbq_u16(inptr2));
    /* Subtract 128 from Cb and Cr. */
    int16x8_t cr_128 = vsubq_n_s16(cr, 128);
    int16x8_t cb_128 = vsubq_n_s16(cb, 128);
    /* Compute Y-G: 0.34414 * (Cb - 128) + 0.71414 * (Cr - 128) */
    int32x4_t y_sub_g_b = vaddq_s32(vqdmullbq_n_s16(cb_128, F_0_344),
                                    vqdmullbq_n_s16(cr_128, F_0_714));
    int32x4_t y_sub_g_t = vaddq_s32(vqdmulltq_n_s16(cb_128, F_0_344),
                                    vqdmulltq_n_s16(cr_128, F_0_714));
    /* Descale G components: shift right 16, round, and narrow to 16-bit. */
    int16x8_t y_sub_g = vrshrnq_n_s32(y_sub_g_b, y_sub_g_t, 16);
    /* Compute R-Y: 1.40200 * (Cr - 128) */
    int16x8_t r_sub_y = vqrdmlahq_n_s16(cr_128, cr_128, F_0_402);
    /* Compute B-Y: 1.77200 * (Cb - 128) */
    int16x8_t b_sub_y = vqrdmlahq_n_s16(cb_128, cb_128, F_0_772);
    /* De-interleave Y component values into two separate vectors, one
     * containing the component values with even-numbered indices and one
     * containing the component values with odd-numbered indices.
     */
    uint8x16_t y = vld1q_u8(inptr0);
    int16x8_t y_even = vreinterpretq_s16_u16(vmovlbq_u8(y));
    int16x8_t y_odd = vreinterpretq_s16_u16(vmovltq_u8(y));
    /* Add the chroma-derived values (Y-G, R-Y, and B-Y) to both the "even" and
     * "odd" Y component values.  This effectively upsamples the chroma
     * components horizontally.
     */
    int16x8_t g_even = vsubq_s16(y_even, y_sub_g);
    int16x8_t r_even = vaddq_s16(y_even, r_sub_y);
    int16x8_t b_even = vaddq_s16(y_even, b_sub_y);
    int16x8_t g_odd  = vsubq_s16(y_odd,  y_sub_g);
    int16x8_t r_odd  = vaddq_s16(y_odd,  r_sub_y);
    int16x8_t b_odd  = vaddq_s16(y_odd,  b_sub_y);
    /* Convert each component to unsigned and narrow, clamping to [0-255].
     * Re-interleave the "even" and "odd" component values.
     */
    uint8x16_t r = vqmovunq_s16(r_even, r_odd);
    uint8x16_t g = vqmovunq_s16(g_even, g_odd);
    uint8x16_t b = vqmovunq_s16(b_even, b_odd);

    mve_pred16_t p = vctp8q(cols_remaining);
#ifdef RGB_ALPHA
    /* Store RGB pixel data to memory. */
    const uint8x16_t offsets = vidupq_n_u8(0, RGB_PIXELSIZE);
    vstrbq_scatter_offset_p_u8(outptr + RGB_RED, offsets, r, p);
    vstrbq_scatter_offset_p_u8(outptr + RGB_GREEN, offsets, g, p);
    vstrbq_scatter_offset_p_u8(outptr + RGB_BLUE, offsets, b, p);
    vstrbq_scatter_offset_p_u8(outptr + RGB_ALPHA, offsets, vdupq_n_u8(0xff), p);
#else
    /* Store RGB pixel data to memory. */
    const uint8x16_t offsets = vmulq_n_u8(vidupq_n_u8(0, 1), RGB_PIXELSIZE);
    vstrbq_scatter_offset_p_u8(outptr + RGB_RED, offsets, r, p);
    vstrbq_scatter_offset_p_u8(outptr + RGB_GREEN, offsets, g, p);
    vstrbq_scatter_offset_p_u8(outptr + RGB_BLUE, offsets, b, p);
#endif
  }
}


/* Upsample and color convert for the case of 2:1 horizontal and 2:1 vertical.
 *
 * See comments above for details regarding color conversion and safe memory
 * access.
 */

void jsimd_h2v2_merged_upsample_helium(JDIMENSION output_width,
                                       JSAMPIMAGE input_buf,
                                       JDIMENSION in_row_group_ctr,
                                       JSAMPARRAY output_buf)
{
  JSAMPROW restrict outptr0, outptr1;
  /* Pointers to Y (both rows), Cb, and Cr data */
  JSAMPROW restrict inptr0_0, inptr0_1, inptr1, inptr2;

  inptr0_0 = input_buf[0][in_row_group_ctr * 2];
  inptr0_1 = input_buf[0][in_row_group_ctr * 2 + 1];
  inptr1 = input_buf[1][in_row_group_ctr];
  inptr2 = input_buf[2][in_row_group_ctr];
  outptr0 = output_buf[0];
  outptr1 = output_buf[1];

  int cols_remaining = output_width;
  for (; cols_remaining >= 16; cols_remaining -= 16) {
    int16x8_t cb = vreinterpretq_s16_u16(vldrbq_u16(inptr1));
    int16x8_t cr = vreinterpretq_s16_u16(vldrbq_u16(inptr2));
    /* Subtract 128 from Cb and Cr. */
    int16x8_t cr_128 = vsubq_n_s16(cr, 128);
    int16x8_t cb_128 = vsubq_n_s16(cb, 128);
    /* Compute Y-G: 0.34414 * (Cb - 128) + 0.71414 * (Cr - 128) */
    int32x4_t y_sub_g_b = vaddq_s32(vqdmullbq_n_s16(cb_128, F_0_344),
                                    vqdmullbq_n_s16(cr_128, F_0_714));
    int32x4_t y_sub_g_t = vaddq_s32(vqdmulltq_n_s16(cb_128, F_0_344),
                                    vqdmulltq_n_s16(cr_128, F_0_714));
    /* Descale G components: shift right 16, round, and narrow to 16-bit. */
    int16x8_t y_sub_g = vrshrnq_n_s32(y_sub_g_b, y_sub_g_t, 16);
    /* Compute R-Y: 1.40200 * (Cr - 128) */
    int16x8_t r_sub_y = vqrdmlahq_n_s16(cr_128, cr_128, F_0_402);
    /* Compute B-Y: 1.77200 * (Cb - 128) */
    int16x8_t b_sub_y = vqrdmlahq_n_s16(cb_128, cb_128, F_0_772);
    /* For each row, de-interleave Y component values into two separate
     * vectors, one containing the component values with even-numbered indices
     * and one containing the component values with odd-numbered indices.
     */
    uint8x16_t y0 = vld1q_u8(inptr0_0);
    uint8x16_t y1 = vld1q_u8(inptr0_1);
    int16x8_t y0_even = vreinterpretq_s16_u16(vmovlbq_u8(y0));
    int16x8_t y0_odd = vreinterpretq_s16_u16(vmovltq_u8(y0));
    int16x8_t y1_even = vreinterpretq_s16_u16(vmovlbq_u8(y1));
    int16x8_t y1_odd = vreinterpretq_s16_u16(vmovltq_u8(y1));
    /* For each row, add the chroma-derived values (G-Y, R-Y, and B-Y) to both
     * the "even" and "odd" Y component values.  This effectively upsamples the
     * chroma components both horizontally and vertically.
     */
    int16x8_t g0_even = vsubq_s16(y0_even, y_sub_g);
    int16x8_t r0_even = vaddq_s16(y0_even, r_sub_y);
    int16x8_t b0_even = vaddq_s16(y0_even, b_sub_y);
    int16x8_t g0_odd  = vsubq_s16(y0_odd,  y_sub_g);
    int16x8_t r0_odd  = vaddq_s16(y0_odd,  r_sub_y);
    int16x8_t b0_odd  = vaddq_s16(y0_odd,  b_sub_y);
    int16x8_t g1_even = vsubq_s16(y1_even, y_sub_g);
    int16x8_t r1_even = vaddq_s16(y1_even, r_sub_y);
    int16x8_t b1_even = vaddq_s16(y1_even, b_sub_y);
    int16x8_t g1_odd  = vsubq_s16(y1_odd,  y_sub_g);
    int16x8_t r1_odd  = vaddq_s16(y1_odd,  r_sub_y);
    int16x8_t b1_odd  = vaddq_s16(y1_odd,  b_sub_y);
    /* Convert each component to unsigned and narrow, clamping to [0-255].
     * Re-interleave the "even" and "odd" component values.
     */
    uint8x16_t r0 = vqmovunq_s16(r0_even, r0_odd);
    uint8x16_t g0 = vqmovunq_s16(g0_even, g0_odd);
    uint8x16_t b0 = vqmovunq_s16(b0_even, b0_odd);
    uint8x16_t r1 = vqmovunq_s16(r1_even, r1_odd);
    uint8x16_t g1 = vqmovunq_s16(g1_even, g1_odd);
    uint8x16_t b1 = vqmovunq_s16(b1_even, b1_odd);

#ifdef RGB_ALPHA
    uint8x16x4_t rgba0, rgba1;
    rgba0.val[RGB_RED] = r0;
    rgba1.val[RGB_RED] = r1;
    rgba0.val[RGB_GREEN] = g0;
    rgba1.val[RGB_GREEN] = g1;
    rgba0.val[RGB_BLUE] = b0;
    rgba1.val[RGB_BLUE] = b1;
    /* Set alpha channel to opaque (0xFF). */
    rgba0.val[RGB_ALPHA] = vdupq_n_u8(0xFF);
    rgba1.val[RGB_ALPHA] = vdupq_n_u8(0xFF);
    /* Store RGBA pixel data to memory. */
    vst4q_u8(outptr0, rgba0);
    vst4q_u8(outptr1, rgba1);
#else
    /* Store RGB pixel data to memory. */
    const uint8x16_t offsets = vmulq_n_u8(vidupq_n_u8(0, 1), RGB_PIXELSIZE);
    vstrbq_scatter_offset_u8(outptr0 + RGB_RED, offsets, r0);
    vstrbq_scatter_offset_u8(outptr0 + RGB_GREEN, offsets, g0);
    vstrbq_scatter_offset_u8(outptr0 + RGB_BLUE, offsets, b0);
    vstrbq_scatter_offset_u8(outptr1 + RGB_RED, offsets, r1);
    vstrbq_scatter_offset_u8(outptr1 + RGB_GREEN, offsets, g1);
    vstrbq_scatter_offset_u8(outptr1 + RGB_BLUE, offsets, b1);
#endif

    /* Increment pointers. */
    inptr0_0 += 16;
    inptr0_1 += 16;
    inptr1 += 8;
    inptr2 += 8;
    outptr0 += (RGB_PIXELSIZE * 16);
    outptr1 += (RGB_PIXELSIZE * 16);
  }

  if (cols_remaining > 0) {
    int16x8_t cb = vreinterpretq_s16_u16(vldrbq_u16(inptr1));
    int16x8_t cr = vreinterpretq_s16_u16(vldrbq_u16(inptr2));
    /* Subtract 128 from Cb and Cr. */
    int16x8_t cr_128 = vsubq_n_s16(cr, 128);
    int16x8_t cb_128 = vsubq_n_s16(cb, 128);
    /* Compute Y-G: 0.34414 * (Cb - 128) + 0.71414 * (Cr - 128) */
    int32x4_t y_sub_g_b = vaddq_s32(vqdmullbq_n_s16(cb_128, F_0_344),
                                    vqdmullbq_n_s16(cr_128, F_0_714));
    int32x4_t y_sub_g_t = vaddq_s32(vqdmulltq_n_s16(cb_128, F_0_344),
                                    vqdmulltq_n_s16(cr_128, F_0_714));
    /* Descale G components: shift right 16, round, and narrow to 16-bit. */
    int16x8_t y_sub_g = vrshrnq_n_s32(y_sub_g_b, y_sub_g_t, 16);
    /* Compute R-Y: 1.40200 * (Cr - 128) */
    int16x8_t r_sub_y = vqrdmlahq_n_s16(cr_128, cr_128, F_0_402);
    /* Compute B-Y: 1.77200 * (Cb - 128) */
    int16x8_t b_sub_y = vqrdmlahq_n_s16(cb_128, cb_128, F_0_772);
    /* For each row, de-interleave Y component values into two separate
     * vectors, one containing the component values with even-numbered indices
     * and one containing the component values with odd-numbered indices.
     */
    uint8x16_t y0 = vld1q_u8(inptr0_0);
    uint8x16_t y1 = vld1q_u8(inptr0_1);
    int16x8_t y0_even = vreinterpretq_s16_u16(vmovlbq_u8(y0));
    int16x8_t y0_odd = vreinterpretq_s16_u16(vmovltq_u8(y0));
    int16x8_t y1_even = vreinterpretq_s16_u16(vmovlbq_u8(y1));
    int16x8_t y1_odd = vreinterpretq_s16_u16(vmovltq_u8(y1));
    /* For each row, add the chroma-derived values (G-Y, R-Y, and B-Y) to both
     * the "even" and "odd" Y component values.  This effectively upsamples the
     * chroma components both horizontally and vertically.
     */
    int16x8_t g0_even = vsubq_s16(y0_even, y_sub_g);
    int16x8_t r0_even = vaddq_s16(y0_even, r_sub_y);
    int16x8_t b0_even = vaddq_s16(y0_even, b_sub_y);
    int16x8_t g0_odd  = vsubq_s16(y0_odd,  y_sub_g);
    int16x8_t r0_odd  = vaddq_s16(y0_odd,  r_sub_y);
    int16x8_t b0_odd  = vaddq_s16(y0_odd,  b_sub_y);
    int16x8_t g1_even = vsubq_s16(y1_even, y_sub_g);
    int16x8_t r1_even = vaddq_s16(y1_even, r_sub_y);
    int16x8_t b1_even = vaddq_s16(y1_even, b_sub_y);
    int16x8_t g1_odd  = vsubq_s16(y1_odd,  y_sub_g);
    int16x8_t r1_odd  = vaddq_s16(y1_odd,  r_sub_y);
    int16x8_t b1_odd  = vaddq_s16(y1_odd,  b_sub_y);
    /* Convert each component to unsigned and narrow, clamping to [0-255].
     * Re-interleave the "even" and "odd" component values.
     */
    uint8x16_t r0 = vqmovunq_s16(r0_even, r0_odd);
    uint8x16_t g0 = vqmovunq_s16(g0_even, g0_odd);
    uint8x16_t b0 = vqmovunq_s16(b0_even, b0_odd);
    uint8x16_t r1 = vqmovunq_s16(r1_even, r1_odd);
    uint8x16_t g1 = vqmovunq_s16(g1_even, g1_odd);
    uint8x16_t b1 = vqmovunq_s16(b1_even, b1_odd);

    mve_pred16_t p = vctp8q(cols_remaining);
#ifdef RGB_ALPHA
    /* Store RGB pixel data to memory. */
    const uint8x16_t offsets = vidupq_n_u8(0, RGB_PIXELSIZE);
    vstrbq_scatter_offset_p_u8(outptr0 + RGB_RED, offsets, r0, p);
    vstrbq_scatter_offset_p_u8(outptr0 + RGB_GREEN, offsets, g0, p);
    vstrbq_scatter_offset_p_u8(outptr0 + RGB_BLUE, offsets, b0, p);
    vstrbq_scatter_offset_p_u8(outptr0 + RGB_ALPHA, offsets, vdupq_n_u8(0xff), p);
    vstrbq_scatter_offset_p_u8(outptr1 + RGB_RED, offsets, r1, p);
    vstrbq_scatter_offset_p_u8(outptr1 + RGB_GREEN, offsets, g1, p);
    vstrbq_scatter_offset_p_u8(outptr1 + RGB_BLUE, offsets, b1, p);
    vstrbq_scatter_offset_p_u8(outptr1 + RGB_ALPHA, offsets, vdupq_n_u8(0xff), p);
#else
    /* Store RGB pixel data to memory. */
    const uint8x16_t offsets = vmulq_n_u8(vidupq_n_u8(0, 1), RGB_PIXELSIZE);
    vstrbq_scatter_offset_p_u8(outptr0 + RGB_RED, offsets, r0, p);
    vstrbq_scatter_offset_p_u8(outptr0 + RGB_GREEN, offsets, g0, p);
    vstrbq_scatter_offset_p_u8(outptr0 + RGB_BLUE, offsets, b0, p);
    vstrbq_scatter_offset_p_u8(outptr1 + RGB_RED, offsets, r1, p);
    vstrbq_scatter_offset_p_u8(outptr1 + RGB_GREEN, offsets, g1, p);
    vstrbq_scatter_offset_p_u8(outptr1 + RGB_BLUE, offsets, b1, p);
#endif
  }
}
