/*
 * jcsample-helium.c - downsampling (Arm Helium)
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

#define JPEG_INTERNALS
#include "../../jinclude.h"
#include "../../jpeglib.h"
#include "../../jsimd.h"
#include "../../jdct.h"
#include "../../jsimddct.h"
#include "../jsimd.h"
#include "../arm/align.h"

#include <arm_mve.h>


/* Downsample pixel values of a single component.
 * This version handles the common case of 2:1 horizontal and 1:1 vertical,
 * without smoothing.
 */

void jsimd_h2v1_downsample_helium(JDIMENSION image_width, int max_v_samp_factor,
                                  JDIMENSION v_samp_factor,
                                  JDIMENSION width_in_blocks,
                                  JSAMPARRAY restrict input_data, JSAMPARRAY restrict output_data)
{
  JSAMPROW restrict inptr, outptr;
  /* Load bias pattern (alternating every pixel.) */
  /* { 0, 1, 0, 1, 0, 1, 0, 1 } */
  const uint16x8_t bias = vreinterpretq_u16_u32(vdupq_n_u32(0x00010000));
  unsigned i, outrow;

  for (outrow = 0; outrow < v_samp_factor; outrow++) {
    outptr = output_data[outrow];
    inptr = input_data[outrow];

    /* Downsample all but the last DCT block of pixels. */
    for (i = 0; i < width_in_blocks - 1; i++) {
      uint8x16_t pixels = vld1q_u8(inptr + i * 2 * DCTSIZE);
      /* Add adjacent pixel values, widen to 16-bit. */
      uint16x8_t samples = vaddq_u16(vmovlbq_u8(pixels), vmovltq_u8(pixels));
      /* Add bias and divide by 2 */
      samples = vhaddq_u16(samples, bias);
      /* Store samples to memory. */
      vstrbq_u16(outptr + i * DCTSIZE, samples);
    }

    /* Load pixels in last DCT block. */
    uint8x16_t pixels = vld1q_u8(inptr + (width_in_blocks - 1) * 2 * DCTSIZE);
    /* Pad the empty elements with the value of the last pixel. */
    uint8_t last_pixel = inptr[image_width - 1];
    int trailing_count = image_width - (width_in_blocks - 1) * 2 * DCTSIZE;
    mve_pred16_t empty_p = vpnot(vctp8q(trailing_count));
    pixels = vdupq_m_n_u8(pixels, last_pixel, empty_p);
    /* Add adjacent pixel values, widen to 16-bit. */
    uint16x8_t samples = vaddq_u16(vmovlbq_u8(pixels), vmovltq_u8(pixels));
    /* Add bias and divide by 2 */
    samples = vhaddq_u16(samples, bias);
    /* Store samples to memory. */
    vstrbq_u16(outptr + (width_in_blocks - 1) * DCTSIZE, samples);
  }
}


/* Downsample pixel values of a single component.
 * This version handles the standard case of 2:1 horizontal and 2:1 vertical,
 * without smoothing.
 */

void jsimd_h2v2_downsample_helium(JDIMENSION image_width, int max_v_samp_factor,
                                  JDIMENSION v_samp_factor,
                                  JDIMENSION width_in_blocks,
                                  JSAMPARRAY restrict input_data,
                                  JSAMPARRAY restrict output_data)
{
  restrict JSAMPROW inptr0, inptr1, outptr;
  /* Load bias pattern (alternating every pixel.) */
  /* { 1, 2, 1, 2, 1, 2, 1, 2 } */
  const uint16x8_t bias = vreinterpretq_u16_u32(vdupq_n_u32(0x00020001));
  unsigned i, outrow;

  for (outrow = 0; outrow < v_samp_factor; outrow++) {
    outptr = output_data[outrow];
    inptr0 = input_data[outrow];
    inptr1 = input_data[outrow + 1];

    /* Downsample all but the last DCT block of pixels. */
    for (i = 0; i < width_in_blocks - 1; i++) {
      uint8x16_t pixels_r0 = vld1q_u8(inptr0 + i * 2 * DCTSIZE);
      uint8x16_t pixels_r1 = vld1q_u8(inptr1 + i * 2 * DCTSIZE);
      /* Add adjacent pixel values in row 0, widen to 16-bit, and add bias. */
      uint16x8_t samples = vaddq_u16(vmovlbq_u8(pixels_r0), vmovltq_u8(pixels_r0));
      samples = vaddq_u16(samples, bias);

      /* Add adjacent pixel values in row 1, widen to 16-bit, and accumulate.
       */
      samples = vaddq_u16(samples, vmovlbq_u8(pixels_r1));
      samples = vaddq_u16(samples, vmovltq_u8(pixels_r1));
      /* Divide total by 4. */
      samples = vshrq_n_u16(samples, 2);
      /* Store samples to memory and increment pointers. */
      vstrbq_u16(outptr + i * DCTSIZE, samples);
    }

    /* Load pixels in last DCT block. */
    uint8x16_t pixels_r0 =
      vld1q_u8(inptr0 + (width_in_blocks - 1) * 2 * DCTSIZE);
    uint8x16_t pixels_r1 =
      vld1q_u8(inptr1 + (width_in_blocks - 1) * 2 * DCTSIZE);
    /* Pad the empty elements with the value of the last pixel. */
    uint8_t last_pixel_r0 = inptr0[image_width - 1];
    uint8_t last_pixel_r1 = inptr1[image_width - 1];
    int trailing_count = image_width - (width_in_blocks - 1) * 2 * DCTSIZE;
    mve_pred16_t empty_p = vpnot(vctp8q(trailing_count));
    pixels_r0 = vdupq_m_n_u8(pixels_r0, last_pixel_r0, empty_p);
    pixels_r1 = vdupq_m_n_u8(pixels_r1, last_pixel_r1, empty_p);
    /* Add adjacent pixel values in row 0, widen to 16-bit, and add bias. */
    uint16x8_t samples = vaddq_u16(vmovlbq_u8(pixels_r0), vmovltq_u8(pixels_r0));
    samples = vaddq_u16(samples, bias);
    /* Add adjacent pixel values in row 1, widen to 16-bit, and accumulate. */
    samples = vaddq_u16(samples, vmovlbq_u8(pixels_r1));
    samples = vaddq_u16(samples, vmovltq_u8(pixels_r1));
    /* Divide total by 4, narrow to 8-bit, and store. */
    samples = vshrq_n_u16(samples, 2);
    vstrbq_u16(outptr + (width_in_blocks - 1) * DCTSIZE, samples);
  }
}
