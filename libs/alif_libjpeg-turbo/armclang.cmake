set(CMAKE_C_COMPILER                armclang)
set(CMAKE_CXX_COMPILER              armclang)
set(CMAKE_C_LINKER_PREFERENCE       armlink)
set(CMAKE_ASM_LINKER_PREFERENCE     armlink)
set(CMAKE_ASM_COMPILER              armasm)
set(CMAKE_ASM_COMPILER_AR           armar)

set(CMAKE_CROSSCOMPILING            true)
set(CMAKE_SYSTEM_NAME               Generic)
set(CMAKE_SYSTEM_PROCESSOR          arm-m)

add_compile_options(
    -mfloat-abi=hard
    --target=arm-arm-none-eabi
    -MD
    -mcpu=cortex-m55)

add_link_options(
    "$<$<CONFIG:RELEASE>:--inline>"
    --tailreorder
    --map
    --symbols
    --info sizes,totals,unused,veneers,summarysizes,inline,tailreorder
    --cpu=cortex-m55)
    
add_compile_definitions(
    NO_GETENV
    NO_PUTENV)