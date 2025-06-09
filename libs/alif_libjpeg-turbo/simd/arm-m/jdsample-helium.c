/*
 * jdsample-helium.c - upsampling (Arm Helium)
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

// Adding the rounding dither is relatively expensive - can speed up
// by undefining this, but we need it on to match the reference implementation.
#define ROUNDING_DITHER

/* The diagram below shows a row of samples produced by h2v1 downsampling.
 *
 *                s0        s1        s2
 *            +---------+---------+---------+
 *            |         |         |         |
 *            | p0   p1 | p2   p3 | p4   p5 |
 *            |         |         |         |
 *            +---------+---------+---------+
 *
 * Samples s0-s2 were created by averaging the original pixel component values
 * centered at positions p0-p5 above.  To approximate those original pixel
 * component values, we proportionally blend the adjacent samples in each row.
 *
 * An upsampled pixel component value is computed by blending the sample
 * containing the pixel center with the nearest neighboring sample, in the
 * ratio 3:1.  For example:
 *     p1(upsampled) = 3/4 * s0 + 1/4 * s1
 *     p2(upsampled) = 3/4 * s1 + 1/4 * s0
 * When computing the first and last pixel component values in the row, there
 * is no adjacent sample to blend, so:
 *     p0(upsampled) = s0
 *     p5(upsampled) = s2
 */

void jsimd_h2v1_fancy_upsample_helium(int max_v_samp_factor,
                                      JDIMENSION downsampled_width,
                                      JSAMPARRAY restrict input_data,
                                      JSAMPARRAY *output_data_ptr)
{
  JSAMPARRAY restrict output_data = *output_data_ptr;
  JSAMPROW inptr, outptr;
  int inrow;
  unsigned colctr;

  for (inrow = 0; inrow < max_v_samp_factor; inrow++) {
    inptr = input_data[inrow];
    outptr = output_data[inrow];
    /* First pixel component value in this row of the original image */
    *outptr = (JSAMPLE)GETJSAMPLE(*inptr);

    /*    3/4 * containing sample + 1/4 * nearest neighboring sample
     * For p1: containing sample = s0, nearest neighboring sample = s1
     * For p2: containing sample = s1, nearest neighboring sample = s0
     */
    uint8x16_t s0 = vld1q_u8(inptr);
    uint8x16_t s1 = vld1q_u8(inptr + 1);
    /* Multiplication makes vectors twice as wide.  '_b' and '_t' suffixes
     * denote bottom half and top half respectively.
     */
    uint16x8_t s1_add_3s0_b = vmlaq_n_u16(vmovlbq_u8(s1), vmovlbq_u8(s0), 3);
    uint16x8_t s1_add_3s0_t = vmlaq_n_u16(vmovltq_u8(s1), vmovltq_u8(s0), 3);
    uint16x8_t s0_add_3s1_b = vmlaq_n_u16(vmovlbq_u8(s0), vmovlbq_u8(s1), 3);
    uint16x8_t s0_add_3s1_t = vmlaq_n_u16(vmovltq_u8(s0), vmovltq_u8(s1), 3);
#ifdef ROUNDING_DITHER
    /* Add ordered dithering bias to odd pixel values. */
    s0_add_3s1_b = vaddq_n_u16(s0_add_3s1_b, 1);
    s0_add_3s1_t = vaddq_n_u16(s0_add_3s1_t, 1);
#endif

    /* The offset is initially 1, because the first pixel component has already
     * been stored.  However, in subsequent iterations of the SIMD loop, this
     * offset is (2 * colctr - 1) to stay within the bounds of the sample
     * buffers without having to resort to a slow scalar tail case for the last
     * (downsampled_width % 16) samples.  See "Creation of 2-D sample arrays"
     * in jmemmgr.c for more details.
     */
    unsigned outptr_offset = 1;
    uint8x16x2_t output_pixels;

    /* We use software pipelining to maximise performance.  The code indented
     * an extra two spaces begins the next iteration of the loop.
     */
    for (colctr = 16; colctr < downsampled_width; colctr += 16) {

        s0 = vld1q_u8(inptr + colctr - 1);
        s1 = vld1q_u8(inptr + colctr);

      /* Right-shift by 2 (divide by 4), narrow to 8-bit, and combine. */
      output_pixels.val[0] = vrshrnq_n_u16(s1_add_3s0_b, s1_add_3s0_t, 2);
#ifdef ROUNDING_DITHER
      output_pixels.val[1] = vshrnq_n_u16(s0_add_3s1_b, s0_add_3s1_t, 2);
#else
      output_pixels.val[1] = vrshrnq_n_u16(s0_add_3s1_b, s0_add_3s1_t, 2);
#endif

        /* Multiplication makes vectors twice as wide.  '_b' and '_t' suffixes
         * denote bottom half and top half respectively.
         */
        s1_add_3s0_b = vmlaq_n_u16(vmovlbq_u8(s1), vmovlbq_u8(s0), 3);
        s1_add_3s0_t = vmlaq_n_u16(vmovltq_u8(s1), vmovltq_u8(s0), 3);
        s0_add_3s1_b = vmlaq_n_u16(vmovlbq_u8(s0), vmovlbq_u8(s1), 3);
        s0_add_3s1_t = vmlaq_n_u16(vmovltq_u8(s0), vmovltq_u8(s1), 3);
#ifdef ROUNDING_DITHER
        /* Add ordered dithering bias to odd pixel values. */
        s0_add_3s1_b = vaddq_n_u16(s0_add_3s1_b, 1);
        s0_add_3s1_t = vaddq_n_u16(s0_add_3s1_t, 1);
#endif

      /* Store pixel component values to memory. */
      vst2q_u8(outptr + outptr_offset, output_pixels);
      outptr_offset = 2 * colctr - 1;
    }

    /* Complete the last iteration of the loop. */

    /* Right-shift by 2 (divide by 4), narrow to 8-bit, and combine. */
    output_pixels.val[0] = vrshrnq_n_u16(s1_add_3s0_b, s1_add_3s0_t, 2);
#ifdef ROUNDING_DITHER
    output_pixels.val[1] = vshrnq_n_u16(s0_add_3s1_b, s0_add_3s1_t, 2);
#else
    output_pixels.val[1] = vrshrnq_n_u16(s0_add_3s1_b, s0_add_3s1_t, 2);
#endif
    /* Store pixel component values to memory. */
    vst2q_u8(outptr + outptr_offset, output_pixels);

    /* Last pixel component value in this row of the original image */
    outptr[2 * downsampled_width - 1] =
      GETJSAMPLE(inptr[downsampled_width - 1]);
  }
}


/* The diagram below shows an array of samples produced by h2v2 downsampling.
 *
 *                s0        s1        s2
 *            +---------+---------+---------+
 *            | p0   p1 | p2   p3 | p4   p5 |
 *       sA   |         |         |         |
 *            | p6   p7 | p8   p9 | p10  p11|
 *            +---------+---------+---------+
 *            | p12  p13| p14  p15| p16  p17|
 *       sB   |         |         |         |
 *            | p18  p19| p20  p21| p22  p23|
 *            +---------+---------+---------+
 *            | p24  p25| p26  p27| p28  p29|
 *       sC   |         |         |         |
 *            | p30  p31| p32  p33| p34  p35|
 *            +---------+---------+---------+
 *
 * Samples s0A-s2C were created by averaging the original pixel component
 * values centered at positions p0-p35 above.  To approximate one of those
 * original pixel component values, we proportionally blend the sample
 * containing the pixel center with the nearest neighboring samples in each
 * row, column, and diagonal.
 *
 * An upsampled pixel component value is computed by first blending the sample
 * containing the pixel center with the nearest neighboring samples in the
 * same column, in the ratio 3:1, and then blending each column sum with the
 * nearest neighboring column sum, in the ratio 3:1.  For example:
 *     p14(upsampled) = 3/4 * (3/4 * s1B + 1/4 * s1A) +
 *                      1/4 * (3/4 * s0B + 1/4 * s0A)
 *                    = 9/16 * s1B + 3/16 * s1A + 3/16 * s0B + 1/16 * s0A
 * When computing the first and last pixel component values in the row, there
 * is no horizontally adjacent sample to blend, so:
 *     p12(upsampled) = 3/4 * s0B + 1/4 * s0A
 *     p23(upsampled) = 3/4 * s2B + 1/4 * s2C
 * When computing the first and last pixel component values in the column,
 * there is no vertically adjacent sample to blend, so:
 *     p2(upsampled) = 3/4 * s1A + 1/4 * s0A
 *     p33(upsampled) = 3/4 * s1C + 1/4 * s2C
 * When computing the corner pixel component values, there is no adjacent
 * sample to blend, so:
 *     p0(upsampled) = s0A
 *     p35(upsampled) = s2C
 */

void jsimd_h2v2_fancy_upsample_helium(int max_v_samp_factor,
                                      JDIMENSION downsampled_width,
                                      JSAMPARRAY restrict input_data,
                                      JSAMPARRAY *output_data_ptr)
{
  JSAMPARRAY output_data = *output_data_ptr;
  JSAMPROW restrict inptr0, inptr1, inptr2, outptr0, outptr1;
  int inrow, outrow;
  unsigned colctr;

  inrow = outrow = 0;
  while (outrow < max_v_samp_factor) {
    inptr0 = input_data[inrow - 1];
    inptr1 = input_data[inrow];
    inptr2 = input_data[inrow + 1];
    /* Suffixes 0 and 1 denote the upper and lower rows of output pixels,
     * respectively.
     */
    outptr0 = output_data[outrow++];
    outptr1 = output_data[outrow++];

    /* We stride by 8 input pixels at a time, reading byte [8N-1..8N+7] and
     * writing [16N-1..16N+14]. The leading edge is handled by vector
     * predication: skip the read of -1 and use 0 instead, and skip the write of -1.
     * The tail edge is handled using a scalar patch-up
     *
     * See the main loop for general comments.
     */

    /* Step 1: Blend samples vertically in columns s0 and s1.
     */

    /* On this leftmost block, we predicate out the read to our left */
    const mve_pred16_t right_7_of_8 = 0xfffc;
    uint16x8_t s0A = vldrbq_z_u16(inptr0 - 1, right_7_of_8);
    uint16x8_t s0B = vldrbq_z_u16(inptr1 - 1, right_7_of_8);
    uint16x8_t s0C = vldrbq_z_u16(inptr2 - 1, right_7_of_8);
    uint16x8_t s1A = vldrbq_u16(inptr0);
    uint16x8_t s1B = vldrbq_u16(inptr1);
    uint16x8_t s1C = vldrbq_u16(inptr2);
    /* Treat s1 as s0 for left-hand column */
    s0A = vpselq_u16(s0A, s1A, right_7_of_8);
    s0B = vpselq_u16(s0B, s1B, right_7_of_8);
    s0C = vpselq_u16(s0C, s1C, right_7_of_8);

    uint16x8_t s0colsum0 = vmlaq_n_u16(s0A, s0B, 3);
    uint16x8_t s0colsum1 = vmlaq_n_u16(s0C, s0B, 3);
    uint16x8_t s1colsum0 = vmlaq_n_u16(s1A, s1B, 3);
    uint16x8_t s1colsum1 = vmlaq_n_u16(s1C, s1B, 3);

    /* Step 2: Blend the already-blended columns. */

    uint16x8_t output0_p1 = vmlaq_n_u16(s1colsum0, s0colsum0, 3);
    uint16x8_t output0_p2 = vmlaq_n_u16(s0colsum0, s1colsum0, 3);
    uint16x8_t output1_p1 = vmlaq_n_u16(s1colsum1, s0colsum1, 3);
    uint16x8_t output1_p2 = vmlaq_n_u16(s0colsum1, s1colsum1, 3);
#ifdef ROUNDING_DITHER
    /* Add ordered dithering bias to odd pixel values. */
    output0_p1 = vaddq_n_u16(output0_p1, 7);
    output1_p1 = vaddq_n_u16(output1_p1, 7);
#endif
    /* Right-shift by 4 (divide by 16), narrow to 8-bit, and combine. */
    uint8x16_t output_pixels0;
#ifdef ROUNDING_DITHER
    output_pixels0 = vshrnbq_n_u16(vuninitializedq_u8(), output0_p1, 4);
    output_pixels0 = vrshrntq_n_u16(output_pixels0, output0_p2, 4);
#else
    output_pixels0 = vrshrnq_n_u16(output0_p1, output0_p2, 4);
#endif
    uint8x16_t output_pixels1;
#ifdef ROUNDING_DITHER
    output_pixels1 = vshrnbq_n_u16(vuninitializedq_u8(), output1_p1, 4);
    output_pixels1 = vrshrntq_n_u16(output_pixels1, output1_p2, 4);
#else
    output_pixels1 = vrshrnq_n_u16(output1_p1, output1_p2, 4);
#endif

    /* Store pixel component values to memory.
     * The minimum size of the output buffer for each row is 64 bytes => no
     * need to worry about buffer overflow here.  See "Creation of 2-D sample
     * arrays" in jmemmgr.c for more details.
     *
     * As with the reads, we predicate out the write to our left.
     */
    const mve_pred16_t right_15_of_16 = 0xfffe;
    vst1q_p_u8(outptr0 - 1, output_pixels0, right_15_of_16);
    vst1q_p_u8(outptr1 - 1, output_pixels1, right_15_of_16);

    /* Continue our aligned strides. We remain aligned to stay within the
     * bounds of the sample buffers.  See "Creation of 2-D sample arrays"
     * in jmemmgr.c for more details.
     */
    for (colctr = 8; colctr < downsampled_width; colctr += 8) {
      /* Step 1: Blend samples vertically in columns s0 and s1.
       * Leave the divide by 4 until the end, when it can be done for both
       * dimensions at once, right-shifting by 4.
       */

      /* Load and compute s0colsum0 and s0colsum1. */
      s0A = vldrbq_u16(inptr0 + colctr - 1);
      s0B = vldrbq_u16(inptr1 + colctr - 1);
      s0C = vldrbq_u16(inptr2 + colctr - 1);
      s0colsum0 = vmlaq_n_u16(s0A, s0B, 3);
      s0colsum1 = vmlaq_n_u16(s0C, s0B, 3);
      /* Load and compute s1colsum0 and s1colsum1. */
      s1A = vldrbq_u16(inptr0 + colctr);
      s1B = vldrbq_u16(inptr1 + colctr);
      s1C = vldrbq_u16(inptr2 + colctr);
      s1colsum0 = vmlaq_n_u16(s1A, s1B, 3);
      s1colsum1 = vmlaq_n_u16(s1C, s1B, 3);

      /* Step 2: Blend the already-blended columns. */

      output0_p1 = vmlaq_n_u16(s1colsum0, s0colsum0, 3);
      output0_p2 = vmlaq_n_u16(s0colsum0, s1colsum0, 3);
      output1_p1 = vmlaq_n_u16(s1colsum1, s0colsum1, 3);
      output1_p2 = vmlaq_n_u16(s0colsum1, s1colsum1, 3);
      /* Add ordered dithering bias to odd pixel values. */
#ifdef ROUNDING_DITHER
      /* (This is a lot of work for little value - just use vrsrhrn for all?) */
      output0_p1 = vaddq_n_u16(output0_p1, 7);
      output1_p1 = vaddq_n_u16(output1_p1, 7);
#endif
      /* Right-shift by 4 (divide by 16), narrow to 8-bit, and combine. */
#ifdef ROUNDING_DITHER
      output_pixels0 = vshrnbq_n_u16(vuninitializedq_u8(), output0_p1, 4);
      output_pixels0 = vrshrntq_n_u16(output_pixels0, output0_p2, 4);
#else
      output_pixels0 = vrshrnq_n_u16(output0_p1, output0_p2, 4);
#endif
#ifdef ROUNDING_DITHER
      output_pixels1 = vshrnbq_n_u16(vuninitializedq_u8(), output1_p1, 4);
      output_pixels1 = vrshrntq_n_u16(output_pixels1, output1_p2, 4);
#else
      output_pixels1 = vrshrnq_n_u16(output1_p1, output1_p2, 4);
#endif
      /* Store pixel component values to memory. */
      vst1q_u8(outptr0 + 2 * colctr - 1, output_pixels0);
      vst1q_u8(outptr1 + 2 * colctr - 1, output_pixels1);
    }

    /* Last pixel component value in this row of the original image */
    int s1colsum0_n = GETJSAMPLE(inptr1[downsampled_width - 1]) * 3 +
                      GETJSAMPLE(inptr0[downsampled_width - 1]);
#ifdef ROUNDING_DITHER
    outptr0[2 * downsampled_width - 1] = (JSAMPLE)((s1colsum0_n + 1) >> 2);
#else
    outptr0[2 * downsampled_width - 1] = (JSAMPLE)urshr(s1colsum0_n, 2);
#endif
    int s1colsum1_n = GETJSAMPLE(inptr1[downsampled_width - 1]) * 3 +
                      GETJSAMPLE(inptr2[downsampled_width - 1]);
#ifdef ROUNDING_DITHER
    outptr1[2 * downsampled_width - 1] = (JSAMPLE)((s1colsum1_n + 1) >> 2);
#else
    outptr1[2 * downsampled_width - 1] = (JSAMPLE)urshr(s1colsum1_n, 2);
#endif
    inrow++;
  }
}


/* The diagram below shows a column of samples produced by h1v2 downsampling
 * (or by losslessly rotating or transposing an h2v1-downsampled image.)
 *
 *            +---------+
 *            |   p0    |
 *     sA     |         |
 *            |   p1    |
 *            +---------+
 *            |   p2    |
 *     sB     |         |
 *            |   p3    |
 *            +---------+
 *            |   p4    |
 *     sC     |         |
 *            |   p5    |
 *            +---------+
 *
 * Samples sA-sC were created by averaging the original pixel component values
 * centered at positions p0-p5 above.  To approximate those original pixel
 * component values, we proportionally blend the adjacent samples in each
 * column.
 *
 * An upsampled pixel component value is computed by blending the sample
 * containing the pixel center with the nearest neighboring sample, in the
 * ratio 3:1.  For example:
 *     p1(upsampled) = 3/4 * sA + 1/4 * sB
 *     p2(upsampled) = 3/4 * sB + 1/4 * sA
 * When computing the first and last pixel component values in the column,
 * there is no adjacent sample to blend, so:
 *     p0(upsampled) = sA
 *     p5(upsampled) = sC
 */

void jsimd_h1v2_fancy_upsample_helium(int max_v_samp_factor,
                                      JDIMENSION downsampled_width,
                                      JSAMPARRAY restrict input_data,
                                      JSAMPARRAY *output_data_ptr)
{
  JSAMPARRAY restrict output_data = *output_data_ptr;
  JSAMPROW inptr0, inptr1, inptr2, outptr0, outptr1;
  int inrow, outrow;
  unsigned colctr;

  inrow = outrow = 0;
  while (outrow < max_v_samp_factor) {
    inptr0 = input_data[inrow - 1];
    inptr1 = input_data[inrow];
    inptr2 = input_data[inrow + 1];
    /* Suffixes 0 and 1 denote the upper and lower rows of output pixels,
     * respectively.
     */
    outptr0 = output_data[outrow++];
    outptr1 = output_data[outrow++];
    inrow++;

    /* The size of the input and output buffers is always a multiple of 32
     * bytes => no need to worry about buffer overflow when reading/writing
     * memory.  See "Creation of 2-D sample arrays" in jmemmgr.c for more
     * details.
     */
    for (colctr = 0; colctr < downsampled_width; colctr += 16) {
      /* Load samples. */
      uint8x16_t sA = vld1q_u8(inptr0 + colctr);
      uint8x16_t sB = vld1q_u8(inptr1 + colctr);
      uint8x16_t sC = vld1q_u8(inptr2 + colctr);
      /* Blend samples vertically. */
      uint16x8_t colsum0_b = vmlaq_n_u16(vmovlbq_u8(sA), vmovlbq_u8(sB), 3);
      uint16x8_t colsum0_t = vmlaq_n_u16(vmovltq_u8(sA), vmovltq_u8(sB), 3);
      uint16x8_t colsum1_b = vmlaq_n_u16(vmovlbq_u8(sC), vmovlbq_u8(sB), 3);
      uint16x8_t colsum1_t = vmlaq_n_u16(vmovltq_u8(sC), vmovltq_u8(sB), 3);
#ifdef ROUNDING_DITHER
      /* Add ordered dithering bias to pixel values in even output rows. */
      colsum0_b = vaddq_n_u16(colsum0_b, 1);
      colsum0_t = vaddq_n_u16(colsum0_t, 1);
#endif
      /* Right-shift by 2 (divide by 4), narrow to 8-bit, and combine. */
      uint16x8_t output_pixels0, output_pixels1;
#ifdef ROUNDING_DITHER
      output_pixels0 = vshrnq_n_u16(colsum0_b, colsum0_t, 2);
#else
      output_pixels0 = vrshrnq_n_u16(colsum0_b, colsum0_t, 2);
#endif
      output_pixels1 = vrshrnq_n_u16(colsum1_b, colsum1_t, 2);
      /* Store pixel component values to memory. */
      vst1q_u8(outptr0 + colctr, output_pixels0);
      vst1q_u8(outptr1 + colctr, output_pixels1);
    }
  }
}


/* The diagram below shows a row of samples produced by h2v1 downsampling.
 *
 *                s0        s1
 *            +---------+---------+
 *            |         |         |
 *            | p0   p1 | p2   p3 |
 *            |         |         |
 *            +---------+---------+
 *
 * Samples s0 and s1 were created by averaging the original pixel component
 * values centered at positions p0-p3 above.  To approximate those original
 * pixel component values, we duplicate the samples horizontally:
 *     p0(upsampled) = p1(upsampled) = s0
 *     p2(upsampled) = p3(upsampled) = s1
 */

void jsimd_h2v1_upsample_helium(int max_v_samp_factor, JDIMENSION output_width,
                                JSAMPARRAY restrict input_data,
                                JSAMPARRAY *output_data_ptr)
{
  JSAMPARRAY restrict output_data = *output_data_ptr;
  JSAMPROW inptr, outptr;
  int inrow;
  unsigned colctr;

  for (inrow = 0; inrow < max_v_samp_factor; inrow++) {
    inptr = input_data[inrow];
    outptr = output_data[inrow];
    for (colctr = 0; 2 * colctr < output_width; colctr += 16) {
      uint8x16_t samples = vld1q_u8(inptr + colctr);
      /* Duplicate the samples.  The store operation below interleaves them so
       * that adjacent pixel component values take on the same sample value,
       * per above.
       */
      uint8x16x2_t output_pixels = { { samples, samples } };
      /* Store pixel component values to memory.
       * Due to the way sample buffers are allocated, we don't need to worry
       * about tail cases when output_width is not a multiple of 32.  See
       * "Creation of 2-D sample arrays" in jmemmgr.c for details.
       */
      vst2q_u8(outptr + 2 * colctr, output_pixels);
    }
  }
}

/* The diagram below shows an array of samples produced by h2v2 downsampling.
 *
 *                s0        s1
 *            +---------+---------+
 *            | p0   p1 | p2   p3 |
 *       sA   |         |         |
 *            | p4   p5 | p6   p7 |
 *            +---------+---------+
 *            | p8   p9 | p10  p11|
 *       sB   |         |         |
 *            | p12  p13| p14  p15|
 *            +---------+---------+
 *
 * Samples s0A-s1B were created by averaging the original pixel component
 * values centered at positions p0-p15 above.  To approximate those original
 * pixel component values, we duplicate the samples both horizontally and
 * vertically:
 *     p0(upsampled) = p1(upsampled) = p4(upsampled) = p5(upsampled) = s0A
 *     p2(upsampled) = p3(upsampled) = p6(upsampled) = p7(upsampled) = s1A
 *     p8(upsampled) = p9(upsampled) = p12(upsampled) = p13(upsampled) = s0B
 *     p10(upsampled) = p11(upsampled) = p14(upsampled) = p15(upsampled) = s1B
 */

void jsimd_h2v2_upsample_helium(int max_v_samp_factor, JDIMENSION output_width,
                                JSAMPARRAY restrict input_data,
                                JSAMPARRAY *output_data_ptr)
{
  JSAMPARRAY restrict output_data = *output_data_ptr;
  JSAMPROW inptr, outptr0, outptr1;
  int inrow, outrow;
  unsigned colctr;

  for (inrow = 0, outrow = 0; outrow < max_v_samp_factor; inrow++) {
    inptr = input_data[inrow];
    outptr0 = output_data[outrow++];
    outptr1 = output_data[outrow++];

    for (colctr = 0; 2 * colctr < output_width; colctr += 16) {
      uint8x16_t samples = vld1q_u8(inptr + colctr);
      /* Duplicate the samples.  The store operation below interleaves them so
       * that adjacent pixel component values take on the same sample value,
       * per above.
       */
      uint8x16x2_t output_pixels = { { samples, samples } };
      /* Store pixel component values for both output rows to memory.
       * Due to the way sample buffers are allocated, we don't need to worry
       * about tail cases when output_width is not a multiple of 32.  See
       * "Creation of 2-D sample arrays" in jmemmgr.c for details.
       */
      vst2q_u8(outptr0 + 2 * colctr, output_pixels);
      vst2q_u8(outptr1 + 2 * colctr, output_pixels);
    }
  }
}
