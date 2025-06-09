/*
 * jsimd_arm.c
 *
 * Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
 * Copyright (C) 2011, Nokia Corporation and/or its subsidiary(-ies).
 * Copyright (C) 2009-2011, 2013-2014, 2016, 2018, 2022, D. R. Commander.
 * Copyright (C) 2015-2016, 2018, 2022, Matthieu Darbois.
 * Copyright (C) 2019, Google LLC.
 * Copyright (C) 2020, Arm Limited.
 * Copyright (C) 2023, Alif Semiconductor.
 *
 * Based on the x86 SIMD extension for IJG JPEG library,
 * Copyright (C) 1999-2006, MIYASAKA Masaru.
 * For conditions of distribution and use, see copyright notice in jsimdext.inc
 *
 * This file contains the interface between the "normal" portions
 * of the library and the SIMD implementations when running on a
 * Arm M-profile architecture with MVE (M-profile Vector Extensions).
 *
 * Unlike the Arm A-profile interfaces, there is no run-time detection of
 * MVE - embedded platforms generally compile for a known CPU, and run-time
 * detection would greatly increase code size. If run-time detection were
 * to be added, it should be a configurable compile-time option.
 */

#define JPEG_INTERNALS
#include "../../jinclude.h"
#include "../../jpeglib.h"
#include "../../jsimd.h"
#include "../../jdct.h"
#include "../../jsimddct.h"
#include "../jsimd.h"

#include <ctype.h>


GLOBAL(int)
jsimd_can_rgb_ycc(void)
{
  /* The code is optimised for these values only */
  if (BITS_IN_JSAMPLE != 8)
    return 0;
  if (sizeof(JDIMENSION) != 4)
    return 0;
  if ((RGB_PIXELSIZE != 3) && (RGB_PIXELSIZE != 4))
    return 0;

  return 1;
}

GLOBAL(int)
jsimd_can_rgb_gray(void)
{
  /* The code is optimised for these values only */
  if (BITS_IN_JSAMPLE != 8)
    return 0;
  if (sizeof(JDIMENSION) != 4)
    return 0;
  if ((RGB_PIXELSIZE != 3) && (RGB_PIXELSIZE != 4))
    return 0;

  return 1;
}

GLOBAL(int)
jsimd_can_ycc_rgb(void)
{
  /* The code is optimised for these values only */
  if (BITS_IN_JSAMPLE != 8)
    return 0;
  if (sizeof(JDIMENSION) != 4)
    return 0;
  if ((RGB_PIXELSIZE != 3) && (RGB_PIXELSIZE != 4))
    return 0;

  return 1;
}

GLOBAL(int)
jsimd_can_ycc_rgb565(void)
{
  /* The code is optimised for these values only */
  if (BITS_IN_JSAMPLE != 8)
    return 0;
  if (sizeof(JDIMENSION) != 4)
    return 0;

  return 1;
}

GLOBAL(void)
jsimd_rgb_ycc_convert(j_compress_ptr cinfo, JSAMPARRAY input_buf,
                      JSAMPIMAGE output_buf, JDIMENSION output_row,
                      int num_rows)
{
  void (*heliumfct) (JDIMENSION, JSAMPARRAY, JSAMPIMAGE, JDIMENSION, int);

  switch (cinfo->in_color_space) {
  case JCS_EXT_RGB:
    heliumfct = jsimd_extrgb_ycc_convert_helium;
    break;
  case JCS_EXT_RGBX:
  case JCS_EXT_RGBA:
    heliumfct = jsimd_extrgbx_ycc_convert_helium;
    break;
  case JCS_EXT_BGR:
    heliumfct = jsimd_extbgr_ycc_convert_helium;
    break;
  case JCS_EXT_BGRX:
  case JCS_EXT_BGRA:
    heliumfct = jsimd_extbgrx_ycc_convert_helium;
    break;
  case JCS_EXT_XBGR:
  case JCS_EXT_ABGR:
    heliumfct = jsimd_extxbgr_ycc_convert_helium;
    break;
  case JCS_EXT_XRGB:
  case JCS_EXT_ARGB:
    heliumfct = jsimd_extxrgb_ycc_convert_helium;
    break;
  default:
    heliumfct = jsimd_extrgb_ycc_convert_helium;
    break;
  }

  heliumfct(cinfo->image_width, input_buf, output_buf, output_row, num_rows);
}

GLOBAL(void)
jsimd_rgb_gray_convert(j_compress_ptr cinfo, JSAMPARRAY input_buf,
                       JSAMPIMAGE output_buf, JDIMENSION output_row,
                       int num_rows)
{
  void (*heliumfct) (JDIMENSION, JSAMPARRAY, JSAMPIMAGE, JDIMENSION, int);

  switch (cinfo->in_color_space) {
  case JCS_EXT_RGB:
    heliumfct = jsimd_extrgb_gray_convert_helium;
    break;
  case JCS_EXT_RGBX:
  case JCS_EXT_RGBA:
    heliumfct = jsimd_extrgbx_gray_convert_helium;
    break;
  case JCS_EXT_BGR:
    heliumfct = jsimd_extbgr_gray_convert_helium;
    break;
  case JCS_EXT_BGRX:
  case JCS_EXT_BGRA:
    heliumfct = jsimd_extbgrx_gray_convert_helium;
    break;
  case JCS_EXT_XBGR:
  case JCS_EXT_ABGR:
    heliumfct = jsimd_extxbgr_gray_convert_helium;
    break;
  case JCS_EXT_XRGB:
  case JCS_EXT_ARGB:
    heliumfct = jsimd_extxrgb_gray_convert_helium;
    break;
  default:
    heliumfct = jsimd_extrgb_gray_convert_helium;
    break;
  }

  heliumfct(cinfo->image_width, input_buf, output_buf, output_row, num_rows);

}

GLOBAL(void)
jsimd_ycc_rgb_convert(j_decompress_ptr cinfo, JSAMPIMAGE input_buf,
                      JDIMENSION input_row, JSAMPARRAY output_buf,
                      int num_rows)
{
  void (*heliumfct) (JDIMENSION, JSAMPIMAGE, JDIMENSION, JSAMPARRAY, int);

  switch (cinfo->out_color_space) {
  case JCS_EXT_RGB:
    heliumfct = jsimd_ycc_extrgb_convert_helium;
    break;
  case JCS_EXT_RGBX:
  case JCS_EXT_RGBA:
    heliumfct = jsimd_ycc_extrgbx_convert_helium;
    break;
  case JCS_EXT_BGR:
    heliumfct = jsimd_ycc_extbgr_convert_helium;
    break;
  case JCS_EXT_BGRX:
  case JCS_EXT_BGRA:
    heliumfct = jsimd_ycc_extbgrx_convert_helium;
    break;
  case JCS_EXT_XBGR:
  case JCS_EXT_ABGR:
    heliumfct = jsimd_ycc_extxbgr_convert_helium;
    break;
  case JCS_EXT_XRGB:
  case JCS_EXT_ARGB:
    heliumfct = jsimd_ycc_extxrgb_convert_helium;
    break;
  default:
    heliumfct = jsimd_ycc_extrgb_convert_helium;
    break;
  }

  heliumfct(cinfo->output_width, input_buf, input_row, output_buf, num_rows);
}

GLOBAL(void)
jsimd_ycc_rgb565_convert(j_decompress_ptr cinfo, JSAMPIMAGE input_buf,
                         JDIMENSION input_row, JSAMPARRAY output_buf,
                         int num_rows)
{
  jsimd_ycc_rgb565_convert_helium(cinfo->output_width, input_buf, input_row,
                                  output_buf, num_rows);
}

GLOBAL(int)
jsimd_can_h2v2_downsample(void)
{
  /* The code is optimised for these values only */
  if (BITS_IN_JSAMPLE != 8)
    return 0;
  if (DCTSIZE != 8)
    return 0;
  if (sizeof(JDIMENSION) != 4)
    return 0;

  return 1;
}

GLOBAL(int)
jsimd_can_h2v1_downsample(void)
{
  /* The code is optimised for these values only */
  if (BITS_IN_JSAMPLE != 8)
    return 0;
  if (DCTSIZE != 8)
    return 0;
  if (sizeof(JDIMENSION) != 4)
    return 0;

  return 1;
}

GLOBAL(void)
jsimd_h2v2_downsample(j_compress_ptr cinfo, jpeg_component_info *compptr,
                      JSAMPARRAY input_data, JSAMPARRAY output_data)
{
  jsimd_h2v2_downsample_helium(cinfo->image_width, cinfo->max_v_samp_factor,
                               compptr->v_samp_factor, compptr->width_in_blocks,
                               input_data, output_data);
}

GLOBAL(void)
jsimd_h2v1_downsample(j_compress_ptr cinfo, jpeg_component_info *compptr,
                      JSAMPARRAY input_data, JSAMPARRAY output_data)
{
  jsimd_h2v1_downsample_helium(cinfo->image_width, cinfo->max_v_samp_factor,
                               compptr->v_samp_factor, compptr->width_in_blocks,
                               input_data, output_data);
}

GLOBAL(int)
jsimd_can_h2v2_upsample(void)
{
  /* The code is optimised for these values only */
  if (BITS_IN_JSAMPLE != 8)
    return 0;
  if (sizeof(JDIMENSION) != 4)
    return 0;

  return 1;
}

GLOBAL(int)
jsimd_can_h2v1_upsample(void)
{
  /* The code is optimised for these values only */
  if (BITS_IN_JSAMPLE != 8)
    return 0;
  if (sizeof(JDIMENSION) != 4)
    return 0;

  return 1;
}

GLOBAL(void)
jsimd_h2v2_upsample(j_decompress_ptr cinfo, jpeg_component_info *compptr,
                    JSAMPARRAY input_data, JSAMPARRAY *output_data_ptr)
{
  jsimd_h2v2_upsample_helium(cinfo->max_v_samp_factor, cinfo->output_width,
                             input_data, output_data_ptr);
}

GLOBAL(void)
jsimd_h2v1_upsample(j_decompress_ptr cinfo, jpeg_component_info *compptr,
                    JSAMPARRAY input_data, JSAMPARRAY *output_data_ptr)
{
  jsimd_h2v1_upsample_helium(cinfo->max_v_samp_factor, cinfo->output_width,
                             input_data, output_data_ptr);
}

GLOBAL(int)
jsimd_can_h2v2_fancy_upsample(void)
{
  /* The code is optimised for these values only */
  if (BITS_IN_JSAMPLE != 8)
    return 0;
  if (sizeof(JDIMENSION) != 4)
    return 0;

  return 1;
}

GLOBAL(int)
jsimd_can_h2v1_fancy_upsample(void)
{
  /* The code is optimised for these values only */
  if (BITS_IN_JSAMPLE != 8)
    return 0;
  if (sizeof(JDIMENSION) != 4)
    return 0;

  return 1;
}

GLOBAL(int)
jsimd_can_h1v2_fancy_upsample(void)
{
  /* The code is optimised for these values only */
  if (BITS_IN_JSAMPLE != 8)
    return 0;
  if (sizeof(JDIMENSION) != 4)
    return 0;

  return 1;
}

GLOBAL(void)
jsimd_h2v2_fancy_upsample(j_decompress_ptr cinfo, jpeg_component_info *compptr,
                          JSAMPARRAY input_data, JSAMPARRAY *output_data_ptr)
{
  jsimd_h2v2_fancy_upsample_helium(cinfo->max_v_samp_factor,
                                   compptr->downsampled_width, input_data,
                                   output_data_ptr);
}

GLOBAL(void)
jsimd_h2v1_fancy_upsample(j_decompress_ptr cinfo, jpeg_component_info *compptr,
                          JSAMPARRAY input_data, JSAMPARRAY *output_data_ptr)
{
  jsimd_h2v1_fancy_upsample_helium(cinfo->max_v_samp_factor,
                                   compptr->downsampled_width, input_data,
                                   output_data_ptr);
}

GLOBAL(void)
jsimd_h1v2_fancy_upsample(j_decompress_ptr cinfo, jpeg_component_info *compptr,
                          JSAMPARRAY input_data, JSAMPARRAY *output_data_ptr)
{
  jsimd_h1v2_fancy_upsample_helium(cinfo->max_v_samp_factor,
                                   compptr->downsampled_width, input_data,
                                   output_data_ptr);
}

GLOBAL(int)
jsimd_can_h2v2_merged_upsample(void)
{
  /* The code is optimised for these values only */
  if (BITS_IN_JSAMPLE != 8)
    return 0;
  if (sizeof(JDIMENSION) != 4)
    return 0;

  return 1;
}

GLOBAL(int)
jsimd_can_h2v1_merged_upsample(void)
{
  /* The code is optimised for these values only */
  if (BITS_IN_JSAMPLE != 8)
    return 0;
  if (sizeof(JDIMENSION) != 4)
    return 0;

  return 1;
}

GLOBAL(void)
jsimd_h2v2_merged_upsample(j_decompress_ptr cinfo, JSAMPIMAGE input_buf,
                           JDIMENSION in_row_group_ctr, JSAMPARRAY output_buf)
{
  void (*heliumfct) (JDIMENSION, JSAMPIMAGE, JDIMENSION, JSAMPARRAY);

  switch (cinfo->out_color_space) {
    case JCS_EXT_RGB:
      heliumfct = jsimd_h2v2_extrgb_merged_upsample_helium;
      break;
    case JCS_EXT_RGBX:
    case JCS_EXT_RGBA:
      heliumfct = jsimd_h2v2_extrgbx_merged_upsample_helium;
      break;
    case JCS_EXT_BGR:
      heliumfct = jsimd_h2v2_extbgr_merged_upsample_helium;
      break;
    case JCS_EXT_BGRX:
    case JCS_EXT_BGRA:
      heliumfct = jsimd_h2v2_extbgrx_merged_upsample_helium;
      break;
    case JCS_EXT_XBGR:
    case JCS_EXT_ABGR:
      heliumfct = jsimd_h2v2_extxbgr_merged_upsample_helium;
      break;
    case JCS_EXT_XRGB:
    case JCS_EXT_ARGB:
      heliumfct = jsimd_h2v2_extxrgb_merged_upsample_helium;
      break;
    default:
      heliumfct = jsimd_h2v2_extrgb_merged_upsample_helium;
      break;
  }

  heliumfct(cinfo->output_width, input_buf, in_row_group_ctr, output_buf);
}

GLOBAL(void)
jsimd_h2v1_merged_upsample(j_decompress_ptr cinfo, JSAMPIMAGE input_buf,
                           JDIMENSION in_row_group_ctr, JSAMPARRAY output_buf)
{
  void (*heliumfct) (JDIMENSION, JSAMPIMAGE, JDIMENSION, JSAMPARRAY);

  switch (cinfo->out_color_space) {
    case JCS_EXT_RGB:
      heliumfct = jsimd_h2v1_extrgb_merged_upsample_helium;
      break;
    case JCS_EXT_RGBX:
    case JCS_EXT_RGBA:
      heliumfct = jsimd_h2v1_extrgbx_merged_upsample_helium;
      break;
    case JCS_EXT_BGR:
      heliumfct = jsimd_h2v1_extbgr_merged_upsample_helium;
      break;
    case JCS_EXT_BGRX:
    case JCS_EXT_BGRA:
      heliumfct = jsimd_h2v1_extbgrx_merged_upsample_helium;
      break;
    case JCS_EXT_XBGR:
    case JCS_EXT_ABGR:
      heliumfct = jsimd_h2v1_extxbgr_merged_upsample_helium;
      break;
    case JCS_EXT_XRGB:
    case JCS_EXT_ARGB:
      heliumfct = jsimd_h2v1_extxrgb_merged_upsample_helium;
      break;
    default:
      heliumfct = jsimd_h2v1_extrgb_merged_upsample_helium;
      break;
  }

  heliumfct(cinfo->output_width, input_buf, in_row_group_ctr, output_buf);
}

GLOBAL(int)
jsimd_can_convsamp(void)
{
  /* The code is optimised for these values only */
  if (DCTSIZE != 8)
    return 0;
  if (BITS_IN_JSAMPLE != 8)
    return 0;
  if (sizeof(JDIMENSION) != 4)
    return 0;
  if (sizeof(DCTELEM) != 2)
    return 0;

  return 1;
}

GLOBAL(int)
jsimd_can_convsamp_float(void)
{
  /* The code is optimised for these values only */
  if (DCTSIZE != 8)
    return 0;
  if (BITS_IN_JSAMPLE != 8)
    return 0;
  if (sizeof(JDIMENSION) != 4)
    return 0;
  if (sizeof(FAST_FLOAT) != 4)
    return 0;

  return (__ARM_FEATURE_MVE & 2) != 0;
}

GLOBAL(void)
jsimd_convsamp(JSAMPARRAY sample_data, JDIMENSION start_col,
               DCTELEM *workspace)
{
  jsimd_convsamp_helium(sample_data, start_col, workspace);
}

GLOBAL(void)
jsimd_convsamp_float(JSAMPARRAY sample_data, JDIMENSION start_col,
                     FAST_FLOAT *workspace)
{
  jsimd_convsamp_float_helium(sample_data, start_col, workspace);
}

GLOBAL(int)
jsimd_can_fdct_islow(void)
{
  /* The code is optimised for these values only */
  if (DCTSIZE != 8)
    return 0;
  if (sizeof(DCTELEM) != 2)
    return 0;

  return 1;
}

GLOBAL(int)
jsimd_can_fdct_ifast(void)
{
  /* The code is optimised for these values only */
  if (DCTSIZE != 8)
    return 0;
  if (sizeof(DCTELEM) != 2)
    return 0;

  return 1;
}

GLOBAL(int)
jsimd_can_fdct_float(void)
{
  /* The code is optimised for these values only */
  if (DCTSIZE != 8)
    return 0;
  if (sizeof(FAST_FLOAT) != 4)
    return 0;

  return (__ARM_FEATURE_MVE & 2) != 0;
}

GLOBAL(void)
jsimd_fdct_islow(DCTELEM *data)
{
  jsimd_fdct_islow_helium(data);
}

GLOBAL(void)
jsimd_fdct_ifast(DCTELEM *data)
{
  jsimd_fdct_ifast_helium(data);
}

GLOBAL(void)
jsimd_fdct_float(FAST_FLOAT *data)
{
#if __ARM_FEATURE_MVE & 2
  jsimd_fdct_float_helium(data);
#endif
}

GLOBAL(int)
jsimd_can_quantize(void)
{
  /* The code is optimised for these values only */
  if (DCTSIZE != 8)
    return 0;
  if (sizeof(JCOEF) != 2)
    return 0;
  if (sizeof(DCTELEM) != 2)
    return 0;

  return 1;
}

GLOBAL(int)
jsimd_can_quantize_float(void)
{
  /* The code is optimised for these values only */
  if (DCTSIZE != 8)
    return 0;
  if (sizeof(JCOEF) != 2)
    return 0;
  if (sizeof(FAST_FLOAT) != 4)
    return 0;

  return (__ARM_FEATURE_MVE & 2) != 0;
}

GLOBAL(void)
jsimd_quantize(JCOEFPTR coef_block, DCTELEM *divisors, DCTELEM *workspace)
{
  jsimd_quantize_helium(coef_block, divisors, workspace);
}

GLOBAL(void)
jsimd_quantize_float(JCOEFPTR coef_block, FAST_FLOAT *divisors,
                     FAST_FLOAT *workspace)
{
  jsimd_quantize_float_helium(coef_block, divisors, workspace);
}

GLOBAL(int)
jsimd_can_idct_2x2(void)
{
  /* The code is optimised for these values only */
  if (DCTSIZE != 8)
    return 0;
  if (sizeof(JCOEF) != 2)
    return 0;
  if (BITS_IN_JSAMPLE != 8)
    return 0;
  if (sizeof(JDIMENSION) != 4)
    return 0;
  if (sizeof(ISLOW_MULT_TYPE) != 2)
    return 0;

  return 1;
}

GLOBAL(int)
jsimd_can_idct_4x4(void)
{
  /* The code is optimised for these values only */
  if (DCTSIZE != 8)
    return 0;
  if (sizeof(JCOEF) != 2)
    return 0;
  if (BITS_IN_JSAMPLE != 8)
    return 0;
  if (sizeof(JDIMENSION) != 4)
    return 0;
  if (sizeof(ISLOW_MULT_TYPE) != 2)
    return 0;
  if (CENTERJSAMPLE != 128)
     return 0;

  return 1;
}

GLOBAL(void)
jsimd_idct_2x2(j_decompress_ptr cinfo, jpeg_component_info *compptr,
               JCOEFPTR coef_block, JSAMPARRAY output_buf,
               JDIMENSION output_col)
{
  jsimd_idct_2x2_helium(compptr->dct_table, coef_block, output_buf,
                        output_col);
}

GLOBAL(void)
jsimd_idct_4x4(j_decompress_ptr cinfo, jpeg_component_info *compptr,
               JCOEFPTR coef_block, JSAMPARRAY output_buf,
               JDIMENSION output_col)
{
  jsimd_idct_4x4_helium(compptr->dct_table, coef_block, output_buf,
                        output_col);
}

GLOBAL(int)
jsimd_can_idct_islow(void)
{
  /* The code is optimised for these values only */
  if (DCTSIZE != 8)
    return 0;
  if (sizeof(JCOEF) != 2)
    return 0;
  if (BITS_IN_JSAMPLE != 8)
    return 0;
  if (sizeof(JDIMENSION) != 4)
    return 0;
  if (sizeof(ISLOW_MULT_TYPE) != 2)
    return 0;

  return 1;
}

GLOBAL(int)
jsimd_can_idct_ifast(void)
{
  /* The code is optimised for these values only */
  if (DCTSIZE != 8)
    return 0;
  if (sizeof(JCOEF) != 2)
    return 0;
  if (BITS_IN_JSAMPLE != 8)
    return 0;
  if (sizeof(JDIMENSION) != 4)
    return 0;
  if (sizeof(IFAST_MULT_TYPE) != 2)
    return 0;
  if (IFAST_SCALE_BITS != 2)
    return 0;

  return 1;
}

GLOBAL(int)
jsimd_can_idct_float(void)
{
  /* The code is optimised for these values only */
  if (DCTSIZE != 8)
    return 0;
  if (sizeof(JCOEF) != 2)
    return 0;
  if (BITS_IN_JSAMPLE != 8)
    return 0;
  if (sizeof(JDIMENSION) != 4)
    return 0;
  if (sizeof(FAST_FLOAT) != 4)
    return 0;

  return (__ARM_FEATURE_MVE & 2) != 0;
}

GLOBAL(void)
jsimd_idct_islow(j_decompress_ptr cinfo, jpeg_component_info *compptr,
                 JCOEFPTR coef_block, JSAMPARRAY output_buf,
                 JDIMENSION output_col)
{
  jsimd_idct_islow_helium(compptr->dct_table, coef_block, output_buf,
                          output_col);
}

GLOBAL(void)
jsimd_idct_ifast(j_decompress_ptr cinfo, jpeg_component_info *compptr,
                 JCOEFPTR coef_block, JSAMPARRAY output_buf,
                 JDIMENSION output_col)
{
  jsimd_idct_ifast_helium(compptr->dct_table, coef_block, output_buf,
                          output_col);
}

GLOBAL(void)
jsimd_idct_float(j_decompress_ptr cinfo, jpeg_component_info *compptr,
                 JCOEFPTR coef_block, JSAMPARRAY output_buf,
                 JDIMENSION output_col)
{
#if __ARM_FEATURE_MVE & 2
  jsimd_idct_float_helium(compptr->dct_table, coef_block, output_buf,
                          output_col);
#endif
}

GLOBAL(int)
jsimd_can_huff_encode_one_block(void)
{
  if (DCTSIZE != 8)
    return 0;
  if (sizeof(JCOEF) != 2)
    return 0;

  return 1;
}

GLOBAL(JOCTET *)
jsimd_huff_encode_one_block(void *state, JOCTET *buffer, JCOEFPTR block,
                            int last_dc_val, c_derived_tbl *dctbl,
                            c_derived_tbl *actbl)
{
  /* With Arm Compiler 6.19, it's better to pass the table in
   * as a parameter - if the compiler can see the constants in
   * jchuff-helium.c, it generates worse code doing
   * full-width PC-relative literal loads.
   */
  static const UINT8 order_table[] = {
    0,  1,  8, 16,  9,  2,  3, 10,
   17, 24, 32, 25, 18, 11,  4,  5,
   12, 19, 26, 33, 40, 48, 41, 34,
   27, 20, 13,  6,  7, 14, 21, 28,
   35, 42, 49, 56, 57, 50, 43, 36,
   29, 22, 15, 23, 30, 37, 44, 51,
   58, 59, 52, 45, 38, 31, 39, 46,
   53, 60, 61, 54, 47, 55, 62, 63,
  };

  return jsimd_huff_encode_one_block_helium(state, buffer, block, last_dc_val,
                                            dctbl, actbl, order_table);
}

GLOBAL(int)
jsimd_can_encode_mcu_AC_first_prepare(void)
{
  if (DCTSIZE != 8)
    return 0;
  if (sizeof(JCOEF) != 2)
    return 0;

  return 1;
}

GLOBAL(void)
jsimd_encode_mcu_AC_first_prepare(const JCOEF *block,
                                  const int *jpeg_natural_order_start, int Sl,
                                  int Al, UJCOEF *values, size_t *zerobits)
{
  jsimd_encode_mcu_AC_first_prepare_helium(block, jpeg_natural_order_start,
                                           Sl, Al, values, zerobits);
}

GLOBAL(int)
jsimd_can_encode_mcu_AC_refine_prepare(void)
{
  if (DCTSIZE != 8)
    return 0;
  if (sizeof(JCOEF) != 2)
    return 0;

  return 1;
}

GLOBAL(int)
jsimd_encode_mcu_AC_refine_prepare(const JCOEF *block,
                                   const int *jpeg_natural_order_start, int Sl,
                                   int Al, UJCOEF *absvalues, size_t *bits)
{
  return jsimd_encode_mcu_AC_refine_prepare_helium(block,
                                                   jpeg_natural_order_start, Sl,
                                                   Al, absvalues, bits);
}
