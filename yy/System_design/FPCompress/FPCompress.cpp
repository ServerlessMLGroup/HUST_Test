#include "DataEncoders/FPCompress/FPCompress.h"

using namespace NVM;

FPCompress::FPCompress()
{
    /* clear statistics */
    p0 = 0;
    p1 = 0;
    p2 = 0;
    p3 = 0;
    p4 = 0;
    p5 = 0;
    p6 = 0;
    p7 = 0;
}

FPCompress::~FPCompress()
{
    /*
     * nothing to do here
     */
}

void FPCompress::SetConfig( Config *config, bool /*createChildren*/ )
{
    Params *params = new Params( );
    params->SetParams( config );
    SetParams( params );
}

void FPCompress::RegisterStats()
{
    AddStat(p0);
    AddStat(p1);
    AddStat(p2);
    AddStat(p3);
    AddStat(p4);
    AddStat(p5);
    AddStat(p6);
    AddStat(p7);
}

ncycle_t FPCompress::Read( NVMainRequest *request )
{
   ncycle_t rv = 0;

    // TODO: Add some energy here

    return rv;
}

ncycle_t FPCompress::Write( NVMainRequest *request ){
    NVMDataBlock& newData = request->data;
    NVMDataBlock& oldData = request->oldData;

    int64_t *ptr = (int64_t*)(newData.rawData);

    uint64_t offset = 0, size =  0, encode_data = 0;

    while(size < newData.GetSize()){
        int64_t dword = *ptr;

        if(pattern_zero(dword)){
            encode_data = 0;
            set_data(oldData.rawData, offset,  encode_data);
            p0++; /* zero statistics */
            offset += 3;
        }else if(pattern_one(dword)){
            encode_data = dword & 0xff;
            encode_data |= (0x1 << 8);
            set_data(oldData.rawData, offset,  encode_data);
            p1++;
            offset += 11;
        }else if(pattern_two(dword)){
            encode_data = dword & 0xffff;
            encode_data |= (0x2 << 16);
            set_data(oldData.rawData, offset,  encode_data);
            p2++;
            offset += 19;
        }else if(pattern_three(dword)){
            encode_data = dword & 0xffffffff;
            encode_data |= ((uint64_t)0x3 << 32);
            set_data(oldData.rawData, offset,  encode_data);
            p3++;
            offset += 35;
        }else if(pattern_four(dword)){
            encode_data = dword >> 32;
            encode_data |= ((uint64_t)0x4 << 32);
            set_data(oldData.rawData, offset,  encode_data);
            p4++;
            offset += 35;
        }else if(pattern_five(dword)){
            encode_data = (dword & 0xffff);
            encode_data |= ((dword >> 32) & 0xffff) << 16;
            encode_data |= ((uint64_t)0x5 << 32);
            set_data(oldData.rawData, offset,  encode_data);
            p5++;
            offset += 35;
        }else if(pattern_six(dword)){
            encode_data = dword && 0xffff;
            encode_data |= (0x6 << 16);
            set_data(oldData.rawData, offset,  encode_data);
            p6++;
            offset += 19;
        }else {
            encode_data = dword;
            set_data(oldData.rawData, offset, (encode_data >> 32) & 0xffffffff);
            set_data(oldData.rawData, offset + 32, (encode_data & 0xffffffff));
            p7++;
            offset += 64;
        }

        /* pointer to next dword */
        ptr++;

        size += 8;
    }
}

void FPCompress::set_data( uint8_t *address, uint64_t offset, uint64_t data)
{
    uint64_t *ptr = (uint64_t*)(address + (offset / 8));
    uint64_t new_data = *ptr;
    uint64_t len = offset - (offset / 8) * 8;
    new_data = (new_data << (64 - len)) >> (64 - len);
    new_data |= (data << len);
    *ptr = new_data;
}
