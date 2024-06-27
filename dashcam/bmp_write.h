/* Copyright (C) 2023 Alif Semiconductor - All Rights Reserved.
 * Use, distribution and modification of this code is permitted under the
 * terms stated in the Alif Semiconductor Software License Agreement
 *
 * You should have received a copy of the Alif Semiconductor Software
 * License Agreement with this file. If not, please write to:
 * contact@alifsemi.com, or visit: https://alifsemi.com/license
 *
 */

/**************************************************************************//**
 * @brief    Write image in BMP format using Azure FileX
 ******************************************************************************/

#ifndef BMP_WRITE_H
#define BMP_WRITE_H
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

#include "fx_api.h"
#include "jpeglib.h"

#define ROW_BUFFER_SIZE (1024 * 64)
static uint8_t rowbuffer[ROW_BUFFER_SIZE] __attribute__((section(".bss.camera_frame_bayer_to_rgb_buf"))) __attribute__((aligned(32)));

#pragma pack(1)
typedef struct
{
    uint8_t id[2];
    uint8_t file_size[4];
    uint8_t reserved[4];
    uint8_t offset[4];
} bmp_file_header_t;

typedef struct
{
    uint8_t header_size[4];
    uint8_t width[4];
    uint8_t height[4];
    uint8_t color_planes[2];
    uint8_t bits_per_pixel[2];
    uint8_t compression[4];
    uint8_t raw_image_size[4];
    uint8_t horizontal_res[4];
    uint8_t vertical_res[4];
    uint8_t number_of_palette_colors[4];
    uint8_t number_of_important_colors[4];
} bmp_info_header_t;
#pragma pack(4)

void write_le32(uint8_t *dst, uint32_t value)
{
    *dst++ = (uint8_t)value;
    *dst++ = (uint8_t)(value >> 8);
    *dst++ = (uint8_t)(value >> 16);
    *dst = (uint8_t)(value >> 24);
}

void write_le16(uint8_t *dst, uint16_t value)
{
    *dst++ = (uint8_t)value;
    *dst++ = (uint8_t)(value >> 8);
}

/**
  \fn           bmp_write
  \brief        initialize host controller
  \param[in]    image_file Readily opened FileX handle
  \param[in]    pixel_data RGB pixel data (3bytes per pixel, no alpha channel)
  \param[in]    width Image width (pixels)
  \param[in]    width Image height (pixels)
  */
bool bmp_write(FX_FILE *image_file, uint8_t *pixel_data, uint32_t width, uint32_t height)
{
    bmp_file_header_t file_header;
    bmp_info_header_t info_header;

    static const uint32_t bytes_per_pixel = 3;
    uint32_t pad_bytes = 4 - ((width * 3) % 4);
    if (pad_bytes == 4) { pad_bytes = 0; }
    const uint32_t row_bytes = width * bytes_per_pixel + pad_bytes;
    file_header.id[0] = 'B';
    file_header.id[1] = 'M';
    write_le32(file_header.file_size, row_bytes * height + sizeof(file_header) + sizeof(info_header));
    write_le32(file_header.reserved, 0);
    write_le32(file_header.offset, sizeof(file_header) + sizeof(info_header));

    write_le32(info_header.header_size, sizeof(info_header));
    write_le32(info_header.width, width);
    write_le32(info_header.height, height);
    write_le16(info_header.color_planes, 1);
    write_le16(info_header.bits_per_pixel, bytes_per_pixel * 8);
    write_le32(info_header.compression, 0);
    write_le32(info_header.raw_image_size, 0);
    write_le32(info_header.horizontal_res, 0);
    write_le32(info_header.number_of_palette_colors, 0);
    write_le32(info_header.number_of_important_colors, 0);

    if (row_bytes > ROW_BUFFER_SIZE) {
        return false;
    }

    // Write BMP file header
    bool ret = fx_file_write(image_file, &file_header, sizeof(file_header)) == FX_SUCCESS;

    // Write BMP info header
    ret = ret && fx_file_write(image_file, &info_header, sizeof(info_header)) == FX_SUCCESS;
    
    // Write BGR pixel data
    const int n_buffered_rows_max = ROW_BUFFER_SIZE / row_bytes;
    int n_buffered_rows = 0;
    uint8_t *row_bgr_ptr = rowbuffer;
    // Write row by row (BMP defaults to bottom row first)
    for (int row = height -1; ret && (row >= 0); row--) {

        // Convert to BGR
        const uint8_t *row_rgb_ptr = &pixel_data[row * width * bytes_per_pixel];
        for (uint32_t col = 0; col < width; col++) {
            *row_bgr_ptr++ = row_rgb_ptr[2];
            *row_bgr_ptr++ = row_rgb_ptr[1];
            *row_bgr_ptr++ = row_rgb_ptr[0];
            row_rgb_ptr += 3;
        }

        for (uint32_t ii = 0; ii < pad_bytes; ii++) {
            *row_bgr_ptr++ = 0;
        }

        n_buffered_rows++;
        // Write out the buffered rows
        if (n_buffered_rows == n_buffered_rows_max) {
            ret = ret && fx_file_write(image_file, rowbuffer, row_bytes * n_buffered_rows) == FX_SUCCESS;
            n_buffered_rows = 0;
            row_bgr_ptr = rowbuffer;
        }
    }
    
    if (n_buffered_rows > 0) {
        ret = ret && fx_file_write(image_file, rowbuffer, row_bytes * n_buffered_rows) == FX_SUCCESS;
    }

    return ret;
}

// TODO: Support writing images bigger than compress buffer (rowbuffer)
bool jpeg_write(FX_FILE *image_file, uint8_t *pixel_data, uint32_t width, uint32_t height)
{
    struct jpeg_error_mgr jerr;
    struct jpeg_compress_struct cinfo;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);

    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;

    jpeg_set_defaults(&cinfo);

    cinfo.dct_method = JDCT_FASTEST;

    jpeg_set_quality(&cinfo, 75, TRUE);

    uint8_t* pjb = rowbuffer;
    uint32_t jbsz = sizeof(rowbuffer);
    jpeg_mem_dest(&cinfo, &pjb, &jbsz);


    jpeg_start_compress(&cinfo, TRUE);

    while (cinfo.next_scanline < cinfo.image_height) {
        JSAMPROW jsamp = pixel_data + (cinfo.next_scanline % cinfo.image_height) * width * 3;
        jpeg_write_scanlines(&cinfo, &jsamp, 1);
    }

    jpeg_finish_compress(&cinfo);
    size_t jpeg_bytes = sizeof(rowbuffer) - cinfo.dest->free_in_buffer;

    return cinfo.err->msg_code == 0 &&
           fx_file_write(image_file, rowbuffer, jpeg_bytes) == FX_SUCCESS;
}

#endif // BMP_WRITE_H
