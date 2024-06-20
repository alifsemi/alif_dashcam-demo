# alif_dashcam-demo
Example application which takes camera snapshots and saves them to SD card

This application is built on VSCode Getting Started Template (alif_vscode-template)
The default hardware is Gen 2 Ensemble DevKit with ARX3A0 camera module

Please follow the template project's [Getting started guide](https://github.com/alifsemi/alif_vscode-template/blob/main/doc/getting_started.md) to set up the environment.

Viewfinder demo can be useful to check for [reference](https://github.com/alifsemi/alif_M55-viewfinder)

## Brief description
The application sets the power management profiles and initializes the camera module and SD card.
After initialization one frame is captured. The raw bayer image data is then processed to RGB format and written to SD card in BMP format.
When write is finished the camera and SD card are uninitialized, the wake up timer (LPTIMER) is set and SoC goes to stop mode waiting for LPTIMER interrupt.
- The frame number/id in filename is an RTC timer timestamp
- Azure FILEX provides the FAT file system implementation.
- A readily formatted SD card is expected. Partition should be smaller than 32GIB
- printf is retargeted to UART
- wake up timer is set for 10s sleep between each frame
- red LED is lit during image processing, green LED is lit during the SD write

## Build notes
- CMSIS Pack version Ensemble 1.1.3
  - **NOTE** The pack needs a minor edit in `sd_host.h` header file!
  - With the default environment you can find the file from `C:\Users\<user>\AppData\Local\arm\packs\AlifSemiconductor\Ensemble\1.1.3\drivers\include\sd_host.h`
  - Set the SD clock to 12.5MHz (defaults to 25MHz)
```
  #define SDMMC_OP_FREQ_SEL                       SDMMC_CLK_12_5MHz_DIV
```
- AzureRTOS pack 1.1.0
- SE toolkit and services v1.0.94
  - Remember to update the system package using SE toolkit updateSystemPackage

## Demo notes
- SD card performance
    - On DevKit board there is no possibility to control SD card power from software
    - This is a DevKit board design limitation, not a SoC HW limitation
    - In a real low power product it should be handled and care must be taken when choosing the SD card (there are low power options)
    - The DevKit board does not support UHS mode on SD cards (no 1.8V IO)
- The default camera is not specifically low power, so in this demo it is shut down while SoC is in stop mode
    - In a low power product there is a possibility to use low power camera and LPCPI block on Alif SoC (instead of the CPI/CSI used by default camera in this demo)     
- A different approach would be to keep the camera controller powered up and keep the driver state in the retentioned TCM memory
    - Use the so called 'fast boot' on HE core
    - This would add some power consumption on the idle phase of the capture loop, but it would allow higher framerates because the core and software initialization would be faster
    - However the stop mode based solution was chosen as first demo because the SD card bandwidth is currently dominating the achievable framerate and in the stop mode the power consumption is minimal.
- TODO
    - JPEG encoding of the frames would dramatically increase the achievable FPS
    - Downscaling the image would also help FPS, or save the image as raw bayer?
    - Investigate the SD card throughput from driver/Azure FILEX point of view (Is there something to optimize?)
    - SRAM1 is kept powered in stop mode due to some problem, an internal issue ticket has been raised to solve this out. (this adds some power consumption in stop mode)
