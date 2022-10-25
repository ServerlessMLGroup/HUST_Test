#ifndef __NVMAIN_FPCOMPRESS_H__
#define __NVMAIN_FPCOMPRESS_H__

#include "src/DataEncoder.h"

namespace NVM {

class FPCompress : public DataEncoder {
  public:
    FPCompress();
    ~FPCompress();

    void SetConfig( Config *config, bool createChildren = true );

    ncycle_t Read( NVMainRequest *request );
    ncycle_t Write( NVMainRequest *request );

    void RegisterStats( );

  private:
    uint64_t p0; /* pattern zero */
    uint64_t p1;
    uint64_t p2;
    uint64_t p3;
    uint64_t p4;
    uint64_t p5;
    uint64_t p6;
    uint64_t p7; /* pattern seven */

    void set_data( uint8_t *address, uint64_t index, uint64_t data);
};

/* zero run */
static inline bool pattern_zero(int64_t x){
  return x == 0;
}

/* 8bit sign extended */
static inline bool pattern_one(int64_t x){
  return (x >> 7) == -1 || (x >> 7) == 0;
}

/* 16 bit sign extended */
static inline bool pattern_two(int64_t x){
  return (x >> 15) == -1 || (x >> 15) == 0;
}

/* halfword sign extended */
static inline bool pattern_three(int64_t x){
  return (x >> 31) == -1 || (x >> 31) == 0;
}

/* halfword padded with a zero halfword */
static inline bool pattern_four(int64_t x){
  return (x << 32) == 0;
}

/* two half words, each a byte sign-extended */
static inline bool pattern_five(int64_t x){
  int32_t high = static_cast<int32_t>(x >> 32);
  int32_t low  = static_cast<int32_t>(x & 0xffffffff);

  bool high_valid = (high >> 15) == -1 || (high >> 15) == 0;
  bool low_valid  = (low >> 15) == -1 || (low >> 15) == 0;

  return high_valid && low_valid;
}

/* word consisting of repeated bytes */
static inline bool pattern_six(int64_t x){
  int64_t total = 0;



int64_t pattern_byte = x & 0xffff;

  total |= pattern_byte;
  total <<= 16;

  total |= pattern_byte;
  total <<= 16;

  total |= pattern_byte;
  total <<= 16;

  total |= pattern_byte;
  total <<= 16;

  return total == x;
}

};

#endif
