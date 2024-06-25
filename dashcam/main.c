/* Copyright (C) 2023-2024 Alif Semiconductor - All Rights Reserved.
 * Use, distribution and modification of this code is permitted under the
 * terms stated in the Alif Semiconductor Software License Agreement
 *
 * You should have received a copy of the Alif Semiconductor Software
 * License Agreement with this file. If not, please write to:
 * contact@alifsemi.com, or visit: https://alifsemi.com/license
 *
 */
#include <time.h>
#include <string.h> //memset

/* ThreadX and FileX Includes */
#include "tx_api.h"
#include "fx_api.h"
#include "fx_sd_driver.h"

#include "RTE_Components.h"
#include CMSIS_device_header

// Needed for sd_uninit()
#include "sd.h"

#include "Driver_GPIO.h"
#include "Driver_CPI.h"    // Camera
#include "board.h"
#include "bmp_write.h"
#include "bayer.h"
#include "image_processing.h"
#include "power_management.h"
#include "sd_pinconf.h"
#include "power.h"

#include "se_services_port.h"

// From color_correction.c
void white_balance(int ml_width, int ml_height, const uint8_t *sp, uint8_t *dp);

// Check if UART trace is disabled
#if !defined(DISABLE_UART_TRACE)
#include <stdio.h>
#include "uart_tracelib.h"

static void uart_callback(uint32_t event)
{
}
#else
#define printf(fmt, ...) (0)
#endif

// Azure RTOS and SD card
#define K (1024)
#define STACK_POOL_SIZE (40*K)
#define SD_STACK_SIZE (10*K)
#define SD_BLK_SIZE 512
TX_THREAD snapshot_thread;
TX_BYTE_POOL StackPool;
unsigned char *p_sdStack = NULL;
/* Buffer for FileX FX_MEDIA sector cache. This must be large enough for at least one sector , which are typically 512 bytes in size. */
UCHAR media_memory[SD_BLK_SIZE*10] __attribute__((section("sd_dma_buf"))) __attribute__((aligned(32)));
FX_MEDIA sd_card;

#define BAYER_FORMAT DC1394_COLOR_FILTER_GRBG

/* Camera Controller Resolution. */
#if RTE_Drivers_CAMERA_SENSOR_ARX3A0
#define CAM_FRAME_WIDTH        (RTE_ARX3A0_CAMERA_SENSOR_FRAME_WIDTH)
#define CAM_FRAME_HEIGHT       (RTE_ARX3A0_CAMERA_SENSOR_FRAME_HEIGHT)
#define RGB_BUFFER_SECTION     ".bss.camera_frame_bayer_to_rgb_buf"
#else
// See the viewfinder project for MT9M114 camera and other camera integration instructions
// https://github.com/alifsemi/alif_M55-viewfinder
#error Unsupported camera
#endif

#define CAM_BYTES_PER_PIXEL 
#define CAM_FRAME_SIZE (CAM_FRAME_WIDTH * CAM_FRAME_HEIGHT)
#define CAM_MPIX (CAM_FRAME_SIZE / 1000000.0f)
#define CAM_FRAME_SIZE_BYTES (CAM_FRAME_SIZE)

static uint8_t camera_buffer[CAM_FRAME_SIZE_BYTES] __attribute__((aligned(32), section(".bss.camera_frame_buf")));
static uint8_t image_buffer[CAM_FRAME_SIZE * RGB_BYTES] __attribute__((aligned(32), section(RGB_BUFFER_SECTION)));
static const uint8_t *get_resize_source_buffer() { return image_buffer; }


/* Camera  Driver instance 0 */
extern ARM_DRIVER_CPI Driver_CPI;
static ARM_DRIVER_CPI *CAMERAdrv = &Driver_CPI;

typedef enum {
    CAM_CB_EVENT_NONE            = 0,
    CAM_CB_EVENT_ERROR           = (1 << 0),
    DISP_CB_EVENT_ERROR          = (1 << 1),
    CAM_CB_EVENT_CAPTURE_STOPPED = (1 << 2)
} CB_EVENTS;

static volatile CB_EVENTS g_cb_events = CAM_CB_EVENT_NONE;

static void camera_callback(uint32_t event)
{
    switch (event)
    {
    case ARM_CPI_EVENT_CAMERA_CAPTURE_STOPPED:
        g_cb_events |= CAM_CB_EVENT_CAPTURE_STOPPED;
        break;
    case ARM_CPI_EVENT_CAMERA_FRAME_HSYNC_DETECTED:
        break;
    case ARM_CPI_EVENT_CAMERA_FRAME_VSYNC_DETECTED:
        break;

    case ARM_CPI_EVENT_ERR_CAMERA_INPUT_FIFO_OVERRUN:
    case ARM_CPI_EVENT_ERR_CAMERA_OUTPUT_FIFO_OVERRUN:
    case ARM_CPI_EVENT_ERR_HARDWARE:
    case ARM_CPI_EVENT_MIPI_CSI2_ERROR:
    default:
        g_cb_events |= CAM_CB_EVENT_ERROR;
        break;
    }
}

int camera_init(void)
{
    int ret = CAMERAdrv->Initialize(camera_callback);
    if(ret != ARM_DRIVER_OK)
    {
        printf("\r\n Error: CAMERA Initialize failed.\r\n");
        return ret;
    }

    /* Power up Camera peripheral */
    ret = CAMERAdrv->PowerControl(ARM_POWER_FULL);
    if(ret != ARM_DRIVER_OK)
    {
        printf("\r\n Error: CAMERA Power Up failed.\r\n");
        return ret;
    }

    /* Control configuration for camera controller */
    ret = CAMERAdrv->Control(CPI_CONFIGURE, 0);
    if(ret != ARM_DRIVER_OK)
    {
        printf("\r\n Error: CPI Configuration failed.\r\n");
        return ret;
    }

    /* Control configuration for camera sensor */
    ret = CAMERAdrv->Control(CPI_CAMERA_SENSOR_CONFIGURE, 0);
    if(ret != ARM_DRIVER_OK)
    {
        printf("\r\n Error: CAMERA SENSOR Configuration failed.\r\n");
        return ret;
    }

    /*Control configuration for camera events */
    ret = CAMERAdrv->Control(CPI_EVENTS_CONFIGURE,
                             ARM_CPI_EVENT_CAMERA_CAPTURE_STOPPED |
                             ARM_CPI_EVENT_ERR_CAMERA_INPUT_FIFO_OVERRUN |
                             ARM_CPI_EVENT_ERR_CAMERA_OUTPUT_FIFO_OVERRUN |
                             ARM_CPI_EVENT_ERR_HARDWARE);
    if(ret != ARM_DRIVER_OK)
    {
        printf("\r\n Error: CAMERA SENSOR Event Configuration failed.\r\n");
        return ret;
    }

    return ret;
}

void clock_init(bool enable)
{
    uint32_t service_error_code = 0;
    /* Enable Clocks */
    uint32_t error_code = SERVICES_clocks_enable_clock(se_services_s_handle, CLKEN_CLK_100M, enable, &service_error_code);
    if(error_code || service_error_code){
        printf("SE: 100MHz clock enable error_code=%u se_error_code=%u\n", error_code, service_error_code);
        return;
    }

    error_code = SERVICES_clocks_enable_clock(se_services_s_handle, CLKEN_HFOSC, enable, &service_error_code);
    if(error_code || service_error_code){
        printf("SE: HFOSC enable error_code=%u se_error_code=%u\n", error_code, service_error_code);
        return;
    }

    error_code = SERVICES_clocks_enable_clock(se_services_s_handle, CLKEN_USB, enable, &service_error_code);
    if(error_code || service_error_code){
        printf("SE: SDMMC 20MHz clock enable error_code=%u se_error_code=%u\n", error_code);
        return;
    }
}


bool save_image(char* filename, uint8_t *pixel_data, uint32_t width, uint32_t height)
{
    uint32_t status = fx_file_create(&sd_card, filename);

    /* Check the create status.  */
    if (status != FX_SUCCESS &&
        status != FX_ALREADY_CREATED) // Allow overwriting previous image
    {
        printf("Failed creating '%s' status=%u\n", filename, status);
        return false;
    }

    FX_FILE image_file;
    status = fx_file_open(&sd_card, &image_file, filename, FX_OPEN_FOR_WRITE);
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

    if (!bmp_write(&image_file, pixel_data, width, height)) {
        printf("Failed writing '%s' in BMP format\n", filename);
        fx_file_close(&image_file);
        return false;
    }

    status = fx_file_close(&image_file);
    if (status != FX_SUCCESS)
    {
        printf("Failed closing '%s' status=%u\n", filename, status);
        return false;
    }

    status = fx_media_flush(&sd_card);
    if (status != FX_SUCCESS)
    {
        printf("Failed media flush status=%u\n", status);
        return false;
    }

    printf("Successfully wrote '%s'\n", filename);
    return true;
}

void snapshot_thread_entry(ULONG args)
{
    int ret = 0;

    printf("camera_init\n");
    ret = camera_init();
    printf("camera_init done, ret=%d\n", ret);

    // Camera needs some warm up time. Otherwise picture is dark
    sys_busy_loop_us(10000);

    // TODO: with higher framerate the rtc value does not work
    int frame_id = get_rtc_value();

    /* Open the SD disk. and initialize SD controller */
    uint32_t fx_media_status = fx_media_open(&sd_card, "SD_DISK", _fx_sd_driver, 0, (VOID *)media_memory, sizeof(media_memory));
    if (fx_media_status != FX_SUCCESS)
    {
        printf("media open fail status = %d...\n", fx_media_status);
    } else {
        printf("media open SUCCESS!\n");
    }

    ret = CAMERAdrv->CaptureFrame(camera_buffer);
    if(ret == ARM_DRIVER_OK) {
        // Wait for capture
        while (!(g_cb_events & CAM_CB_EVENT_CAPTURE_STOPPED)) {
            __WFE();
        }
        BOARD_LED1_Control(BOARD_LED_STATE_HIGH); // Turn LED1 on during camera processing
        SCB_CleanInvalidateDCache();
        dc1394_bayer_Simple(camera_buffer, image_buffer, CAM_FRAME_WIDTH, CAM_FRAME_HEIGHT, BAYER_FORMAT);
        white_balance(CAM_FRAME_WIDTH, CAM_FRAME_HEIGHT, image_buffer, image_buffer);

        BOARD_LED1_Control(BOARD_LED_STATE_LOW);
        printf("got frame id=%u\n", frame_id);
        
        if (fx_media_status == FX_SUCCESS) {
            char filename[32];
            snprintf(filename, 32, "frame_%06u.bmp", frame_id);
            BOARD_LED2_Control(BOARD_LED_STATE_HIGH);
            save_image(filename, image_buffer, CAM_FRAME_WIDTH, CAM_FRAME_HEIGHT);
            BOARD_LED2_Control(BOARD_LED_STATE_LOW);
        }
    }

    if (fx_media_status == FX_SUCCESS) {
        printf("close SD\n");
        fx_media_close(&sd_card);
    }

    if (sd_uninit(0) != SD_DRV_STATUS_OK) {
        printf("SD uninit failed\n");
    }

    printf("Prepare for stop\n");
    CAMERAdrv->Stop();
    CAMERAdrv->PowerControl(ARM_POWER_OFF);
    CAMERAdrv->Uninitialize();
 
    clock_init(false);

    // Sleep 10s
    set_wakeup_timer(10000);

    while(1) {
        __disable_irq();
        pm_core_enter_deep_sleep_request_subsys_off();
        __enable_irq();
        printf("ERROR: did not stop --> retry\n");
        sys_busy_loop_us(100000);
    }
}
  
void tx_application_define(void *first_unused_memory){

    /* Tasks memory allocation and creation */
    tx_byte_pool_create(&StackPool, "Stack_pool", first_unused_memory, STACK_POOL_SIZE);

    tx_byte_allocate(&StackPool, (void **)&p_sdStack, SD_STACK_SIZE, TX_NO_WAIT);
    tx_thread_create(&snapshot_thread, "snapshot_thread", snapshot_thread_entry, NULL, p_sdStack, SD_STACK_SIZE,
                     1, 1, TX_NO_TIME_SLICE, TX_AUTO_START);

    /* FileX Initialization */
    fx_system_initialize();

    return;
}


int main (void)
{
    BOARD_Pinmux_Init();

    set_SD_card_pinconf();

    PM_RESET_STATUS last_reset_reason = pm_get_subsystem_reset_status();

    /* Initialize the SE services */
    se_services_port_init();

    uint32_t service_error_code = 0;
    run_profile_t runprof;
    memset(&runprof, 0, sizeof(runprof));
    SERVICES_get_run_cfg(se_services_s_handle, &runprof, &service_error_code);

    bool pm_ok = init_power_management();

    clock_init(true);

#if !defined(DISABLE_UART_TRACE)
    tracelib_init(NULL, uart_callback);
#endif

    printf("Reset reason=%08X rtc=%08X\n",
           last_reset_reason, get_rtc_value());
    printf("pm_ok=%d\n", last_reset_reason, (int)pm_ok);
    
    printf("--get run profile (boot, before set)--\n");
    print_runprofile(&runprof);
    SERVICES_get_run_cfg(se_services_s_handle, &runprof, &service_error_code);

    printf("--get run profile after setting the profile--\n");
    print_runprofile(&runprof);

    tx_kernel_enter();
    return 0;
}
