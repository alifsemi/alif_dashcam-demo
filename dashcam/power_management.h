/* Copyright (C) 2024 Alif Semiconductor - All Rights Reserved.
 * Use, distribution and modification of this code is permitted under the
 * terms stated in the Alif Semiconductor Software License Agreement
 *
 * You should have received a copy of the Alif Semiconductor Software
 * License Agreement with this file. If not, please write to:
 * contact@alifsemi.com, or visit: https://alifsemi.com/license
 *
 */

#ifndef POWER_MANAGEMENT_H_
#define POWER_MANAGEMENT_H_

#include <stdint.h>
#include <stdbool.h>

#include "aipm.h"

void print_runprofile(const run_profile_t *runprof);
uint32_t set_power_profiles(void);
bool init_power_management(void);
int set_wakeup_timer(uint32_t  timeout_ms);
uint32_t get_rtc_value();

#endif