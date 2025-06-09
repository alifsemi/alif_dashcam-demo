#include <arm_mve.h>

/* Combining bottom and top is quite annoying to write, so
 * lets have some wrapper functions, noting special cases
 * where you can use reinterpret rather than a symmetric
 * pair of instructions
 */

/* Don't need the VMOVNB in a VMOVN[B|T] pair */
static INLINE
uint8x16_t vmovnq_u16(uint16x8_t b, uint16x8_t t)
{
  return vmovntq_u16(vreinterpretq_u8_u16(b), t);
}

static INLINE
uint16x8_t vmovnq_u32(uint32x4_t b, uint32x4_t t)
{
  return vmovntq_u32(vreinterpretq_u16_u32(b), t);
}

static INLINE
uint8x16_t vqmovnq_u16(uint16x8_t b, uint16x8_t t)
{
  return vqmovntq_u16(vqmovnbq_u16(vuninitializedq_u8(), b), t);
}

static INLINE
uint8x16_t vqmovunq_s16(int16x8_t b, int16x8_t t)
{
  return vqmovuntq_s16(vqmovunbq_s16(vuninitializedq_u8(), b), t);
}

/* Inline functions don't work for ARM Compiler for intrinsics needing constants */
#define vshrnq_n_u16(b, t, n) \
  vshrntq_n_u16(vshrnbq_n_u16(vuninitializedq_u8(), b, n), \
                t, n)

#define vshrnq_n_u32(b, t, n) \
  vshrntq_n_u32(vshrnbq_n_u32(vuninitializedq_u16(), b, n), \
                t, n)

#define vrshrnq_n_u16(b, t, n) \
  vrshrntq_n_u16(vrshrnbq_n_u16(vuninitializedq_u8(), b, n), \
                 t, n)

#define vrshrnq_n_u32(b, t, n) \
  vrshrntq_n_u32(vrshrnbq_n_u32(vuninitializedq_u16(), b, n), \
                 t, n)

#define vrshrnq_n_s32(b, t, n) \
  vrshrntq_n_s32(vrshrnbq_n_s32(vuninitializedq_s16(), b, n), \
                 t, n)

/* Don't need the VSHRNT in a VSHRN[B|T] #16 pair */
#define vshrnq_16_s32(b, t) \
  vshrnbq_n_s32(vreinterpretq_s16_s32(t), b, 16)

#define vqshrnq_n_s16(b, t, n) \
  vqshrntq_n_s16(vqshrnbq_n_s16(vuninitializedq_s8(), b, n), \
                 t, n)

#define vqrshrnq_n_s16(b, t, n) \
  vqrshrntq_n_s16(vqrshrnbq_n_s16(vuninitializedq_s8(), b, n), \
                  t, n)
