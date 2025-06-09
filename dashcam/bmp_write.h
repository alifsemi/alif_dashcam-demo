/* Copyright (C) 2023-2025 Alif Semiconductor - All Rights Reserved.
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

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

#include "fx_api.h"
#include "jpeglib.h"

#define ROW_BUFFER_SIZE (1024 * 64)
static uint8_t rowbuffer[ROW_BUFFER_SIZE] __attribute__((section("sd_dma_buf"))) __attribute__((aligned(32)));

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

// Callback functions for libjpeg to write using FILEX
static void jpeg_write_callback_init_destination(j_compress_ptr cinfo)
{
    cinfo->dest->next_output_byte = rowbuffer;
    cinfo->dest->free_in_buffer = sizeof(rowbuffer);
}

static boolean jpeg_write_callback_empty_output_buffer(j_compress_ptr cinfo)
{
    FX_FILE *image_file = (FX_FILE *)cinfo->client_data;
    int ret = fx_file_write(image_file, rowbuffer, sizeof(rowbuffer)) == FX_SUCCESS ? 1 : 0;
    cinfo->dest->next_output_byte = rowbuffer;
    cinfo->dest->free_in_buffer = sizeof(rowbuffer);
    return ret;
}

static void jpeg_write_callback_term_destination(j_compress_ptr cinfo)
{
    FX_FILE *image_file = (FX_FILE *)cinfo->client_data;
    fx_file_write(image_file, rowbuffer, sizeof(rowbuffer) - cinfo->dest->free_in_buffer);
}

/**
 \fn           jpeg_write
\param[in]    image_file Readily opened FileX handle
\param[in]    pixel_data RGB pixel data (3bytes per pixel, no alpha channel)
\param[in]    width Image width (pixels)
\param[in]    width Image height (pixels)
*/
bool jpeg_write(FX_FILE *image_file, uint8_t *pixel_data, uint32_t width, uint32_t height)
{
    struct jpeg_destination_mgr dest_mgr;
    struct jpeg_error_mgr jerr;
    struct jpeg_compress_struct cinfo;
    memset(&dest_mgr, 0, sizeof(dest_mgr));
    memset(&jerr, 0, sizeof(jerr));
    memset(&cinfo, 0, sizeof(cinfo));

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);

    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;

    jpeg_set_defaults(&cinfo);

    cinfo.dct_method = JDCT_FLOAT;
    jpeg_set_quality(&cinfo, 85, TRUE);

    dest_mgr.init_destination = &jpeg_write_callback_init_destination;
    dest_mgr.empty_output_buffer = &jpeg_write_callback_empty_output_buffer;
    dest_mgr.term_destination = &jpeg_write_callback_term_destination;
    cinfo.dest = &dest_mgr;
    cinfo.client_data = (void*)image_file;

    jpeg_start_compress(&cinfo, TRUE);

    while (cinfo.next_scanline < cinfo.image_height) {
        JSAMPROW jsamp = pixel_data + (cinfo.next_scanline % cinfo.image_height) * width * 3;
        jpeg_write_scanlines(&cinfo, &jsamp, 1);
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);

    return cinfo.err->msg_code == 0;
}

typedef enum {
    ImageFormat_BMP,
    ImageFormat_JPEG
} ImageFormat;

bool save_image(FX_MEDIA* sd_card, char* filename, ImageFormat format, uint8_t *pixel_data, uint32_t width, uint32_t height)
{
    uint32_t status = fx_file_create(sd_card, filename);

    /* Check the create status.  */
    if (status != FX_SUCCESS &&
        status != FX_ALREADY_CREATED) // Allow overwriting previous image
    {
        printf("Failed creating '%s' status=%u\n", filename, status);
        return false;
    }

    FX_FILE image_file;
    status = fx_file_open(sd_card, &image_file, filename, FX_OPEN_FOR_WRITE);
    if (status != FX_SUCCESS)
    {
        printf("Failed opening '%s' status=%u\n", filename, status);
        return false;
    }

    status = fx_file_seek(&image_file, 0);
    if (status != FX_SUCCESS)
    {
        printf("Failed seeking '%s' status=%u\n", filename, status);
        return false;
    }

    if (format == ImageFormat_BMP) {
        if (!bmp_write(&image_file, pixel_data, width, height)) {
            printf("Failed writing '%s' in BMP format\n", filename);
            fx_file_close(&image_file);
            return false;
        }
    } else if (format == ImageFormat_JPEG) {
        if (!jpeg_write(&image_file, pixel_data, width, height)) {
            printf("Failed writing '%s' in JPEG format\n", filename);
            fx_file_close(&image_file);
            return false;
        }
    } else {
        printf("Unknown image format\n");
        return false;
    }

    status = fx_file_close(&image_file);
    if (status != FX_SUCCESS)
    {
        printf("Failed closing '%s' status=%u\n", filename, status);
        return false;
    }

    status = fx_media_flush(sd_card);
    if (status != FX_SUCCESS)
    {
        printf("Failed media flush status=%u\n", status);
        return false;
    }

    printf("Successfully wrote '%s'\n", filename);
    return true;
}

#ifdef __cplusplus
}
#endif

#endif // BMP_WRITE_H 