/* Copyright (C) 2024 Alif Semiconductor - All Rights Reserved.
 * Use, distribution and modification of this code is permitted under the
 * terms stated in the Alif Semiconductor Software License Agreement
 *
 * You should have received a copy of the Alif Semiconductor Software
 * License Agreement with this file. If not, please write to:
 * contact@alifsemi.com, or visit: https://alifsemi.com/license
 *
 */
#include <stdio.h>
#include <string.h>

#include "power_management.h"
#include "services_lib_api.h"
#include "services_lib_bare_metal.h"

#include "Driver_LPTIMER.h"
#include "Driver_RTC.h"

#define PM_DCDC_VOLTAGE (825)

// Wake up using LPTIMER
#define LPTIMER_CHANNEL    0
extern ARM_DRIVER_LPTIMER DRIVER_LPTIMER0;
static ARM_DRIVER_LPTIMER *lptimerDrv = &DRIVER_LPTIMER0;

// RTC can be used for wake up as well
extern ARM_DRIVER_RTC Driver_RTC0;
static ARM_DRIVER_RTC *RTCdrv = &Driver_RTC0;

static void rtc_callback(uint32_t event)
{
}
// SE handle
extern uint32_t se_services_s_handle;


void print_runprofile(const run_profile_t *runprof)
{
    printf("memory_blocks   %08X\n", runprof->memory_blocks);
    printf("power_domains   %08X\n", runprof->power_domains);
    printf("ip_clock_gating %08X\n", runprof->ip_clock_gating);
    printf("phy_pwr_gating  %08X\n", runprof->phy_pwr_gating);
    printf("vdd_ioflex_3V3  %08X\n", runprof->vdd_ioflex_3V3);
    printf("run_clk_src     %08X\n", runprof->run_clk_src);
    printf("aon_clk_src     %08X\n", runprof->aon_clk_src);
    printf("scaled_clk_freq %08X\n", runprof->scaled_clk_freq);
    printf("dcdc_mode       %08X\n", runprof->dcdc_mode);
    printf("dcdc_voltage    %u\n", runprof->dcdc_voltage);   
}

uint32_t set_power_profiles(void)
{
    run_profile_t default_runprof;
    off_profile_t default_offprof;

    memset(&default_runprof, 0, sizeof(default_runprof));
    memset(&default_offprof, 0, sizeof(default_offprof));

    uint32_t service_error_code;
    uint32_t err = 0;

    default_runprof.power_domains   = PD_VBAT_AON_MASK | PD_SSE700_AON_MASK | PD_SYST_MASK | PD_SESS_MASK;
    default_runprof.power_domains   |= PD_DBSS_MASK;// debug PD
    default_runprof.dcdc_mode       = DCDC_MODE_PWM;
    default_runprof.dcdc_voltage    = PM_DCDC_VOLTAGE;
    default_runprof.aon_clk_src     = CLK_SRC_LFXO;
    default_runprof.run_clk_src     = CLK_SRC_PLL;
    default_runprof.scaled_clk_freq = SCALED_FREQ_XO_HIGH_DIV_38_4_MHZ;
    default_runprof.memory_blocks   = SERAM_MASK | SRAM0_MASK | SRAM1_MASK | MRAM_MASK | FWRAM_MASK;
    default_runprof.ip_clock_gating = LP_PERIPH_MASK;
    default_runprof.ip_clock_gating |= CAMERA_MASK | MIPI_DSI_MASK | MIPI_CSI_MASK | SDC_MASK;
    default_runprof.phy_pwr_gating  |= LDO_PHY_MASK | MIPI_TX_DPHY_MASK | MIPI_RX_DPHY_MASK | MIPI_PLL_DPHY_MASK;

    // For display
    //default_runprof.ip_clock_gating |= CDC200_MASK;

    default_runprof.vdd_ioflex_3V3  = IOFLEX_LEVEL_1V8;

    // for SE 1.0.94 
    default_runprof.ewic_cfg        = EWIC_VBAT_GPIO | EWIC_VBAT_TIMER;
    default_runprof.wakeup_events   = WE_LPGPIO | WE_LPTIMER;
#ifdef CORE_M55_HP
    default_runprof.cpu_clk_freq    = CLOCK_FREQUENCY_400MHZ;
    // for SE 1.0.94 
    default_runprof.vtor_address    = 0x80100000;
    default_runprof.vtor_address_ns = 0x80100000;
#elif CORE_M55_HE
    default_runprof.cpu_clk_freq    = CLOCK_FREQUENCY_160MHZ;
    default_runprof.vtor_address    = 0x80000000;
    default_runprof.vtor_address_ns = 0x80000000;
#else
#error Unsupported core
#endif

    err = SERVICES_set_run_cfg(se_services_s_handle, &default_runprof, &service_error_code);

    if ((err + service_error_code) == 0) {
        default_offprof.power_domains   = 0;
        default_offprof.aon_clk_src     = CLK_SRC_LFXO;
        default_offprof.dcdc_voltage    = PM_DCDC_VOLTAGE;
        default_offprof.dcdc_mode       = DCDC_MODE_PWM;
        default_offprof.stby_clk_src    = CLK_SRC_HFRC;
        default_offprof.stby_clk_freq   = SCALED_FREQ_RC_STDBY_38_4_MHZ;
        // default_offprof.sysref_clk_src = /* SoC Reference Clock shared with all subsystems */
        default_offprof.memory_blocks   = SERAM_MASK | MRAM_MASK;
        default_offprof.memory_blocks   |= SRAM1_MASK;
        default_offprof.ip_clock_gating = LP_PERIPH_MASK;
        default_offprof.phy_pwr_gating  = 0;
        default_offprof.vdd_ioflex_3V3  = IOFLEX_LEVEL_1V8;
        default_offprof.wakeup_events   = WE_LPGPIO | WE_LPTIMER;
        default_offprof.ewic_cfg        = EWIC_VBAT_GPIO | EWIC_VBAT_TIMER;
#ifdef CORE_M55_HP
        default_offprof.vtor_address    = 0x80100000;
        default_offprof.vtor_address_ns = 0x80100000;
#elif CORE_M55_HE
        default_offprof.vtor_address    = 0x80000000;
        default_offprof.vtor_address_ns = 0x80000000;
#else
#error Unsupported core
#endif
        err = SERVICES_set_off_cfg(se_services_s_handle, &default_offprof, &service_error_code);
    }

    return err + service_error_code;
}


static void wakeup_timer_callback(uint8_t event)
{
    int32_t ret = 0;

    if(event == ARM_LPTIMER_EVENT_UNDERFLOW)
    {
        ret = lptimerDrv->Stop(LPTIMER_CHANNEL);
        if(ret != ARM_DRIVER_OK)
        {
            printf("ERROR: Failed to Stop channel %d\n", LPTIMER_CHANNEL);
        }

    }

    return;
}

int32_t init_wakeup_timer(void)
{
    int32_t              ret = -1;

    /* Initialize LPTIMER driver */
    ret = lptimerDrv->Initialize(LPTIMER_CHANNEL, wakeup_timer_callback);
    if(ret != ARM_DRIVER_OK)
    {
        printf("\r\n Error: LPTIMER init failed\r\n");
        return ret;
    }

    /* Enable the power for LPTIMER */
    ret = lptimerDrv->PowerControl(LPTIMER_CHANNEL, ARM_POWER_FULL);
    if(ret != ARM_DRIVER_OK)
    {
        printf("\r\n Error: LPTIMER Power up failed\n");
        return ret;
    }

    return ret;
}

int set_wakeup_timer(uint32_t  timeout_ms)
{
    /*
     *Configuring the lptimer channel for the timeout in seconds
     *Clock Source depends on RTE_LPTIMER_CHANNEL_CLK_SRC in RTE_Device.h
     *RTE_LPTIMER_CHANNEL_CLK_SRC = 0 : 32.768KHz freq (Default)
    */

    int32_t count = timeout_ms * 32768 / 1000;

    /**< Loading the counter value >*/
    int32_t ret = lptimerDrv->Control(LPTIMER_CHANNEL, ARM_LPTIMER_SET_COUNT1, &count);
    if(ret != ARM_DRIVER_OK)
    {
        printf("ERROR: channel '%d'failed to load count\r\n", LPTIMER_CHANNEL);
        return ret;
    }

    ret = lptimerDrv->Start(LPTIMER_CHANNEL);
    if(ret != ARM_DRIVER_OK)
    {
        printf("ERROR: failed to start channel '%d'\r\n", LPTIMER_CHANNEL);
        return ret;
    }

    return ret;
}

int init_rtc()
{
    int ret = RTCdrv->Initialize(rtc_callback);
    if(ret != ARM_DRIVER_OK){
        return ret;
    }

    ret = RTCdrv->PowerControl(ARM_POWER_FULL);
    return ret;
}

uint32_t get_rtc_value()
{
    uint32_t val = 0;
    RTCdrv->ReadCounter(&val);
    return val;
}

bool init_power_management(void)
{
    SERVICES_synchronize_with_se(se_services_s_handle);
    return !init_wakeup_timer() && !init_rtc() && !set_power_profiles();
}

