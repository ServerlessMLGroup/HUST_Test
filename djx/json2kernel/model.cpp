#include <cstddef>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <assert.h>
#include "json.h"
#include "log.h"
#include "model.h"

Model* Model::from_json(const char* json_file) {
    log("enter Model::from_json");
    std::ifstream fs(json_file);
    printf("open file %s status: %d\n", json_file, fs.good());
    std::string tmp, str = "";
    log("ifstream finish");
    while (getline(fs, tmp)) str += tmp;
    fs.close();
    log("getline finish");
    JsonObject* jobj = JsonParser::parse(str);
    log("JsonParser::parse finish");

    Model* m = new Model;
    for (auto sinfo : jobj->mval["storage"]->lval) {
        m->storage.push_back(StorageInfo{
            sinfo->mval["name"]->sval,
            sinfo->mval["size"]->ival,
            sinfo->mval["stype"]->sval
        });
        // printf("name:%s, size:%d, stype:%s\n", sinfo->mval["name"]->sval, sinfo->mval["size"]->ival, sinfo->mval["stype"]->sval);
    }
    printf("finish parse storage!\n");

    for (auto kinfo : jobj->mval["kernels"]->lval) {
        KernelInfo k;

        k.name = kinfo->mval["name"]->sval;
        for (auto arg : kinfo->mval["args"]->lval)
            k.args.push_back(arg->ival);
        
        assert(kinfo->mval["launch_params"]->lval.size() == 6);
        for (int i = 0; i < 6; i++) 
            k.launch_params[i] = kinfo->mval["launch_params"]->lval[i]->ival;
            
        m->kernels.push_back(k);
    }

    printf("finish parse kernels!\n");

    for (auto arg : jobj->mval["args"]->lval) {
        m->args.push_back(arg->ival);
    }

    printf("finish parse args!\n");

    for (auto shared_memory : jobj->mval["shared_memory"]->mval) {
        m->shared_memory[shared_memory.first] = shared_memory.second->ival;
    }
    printf("finish parse shared_memory!\n");

    delete jobj;

    return m;
}

size_t Model::get_stype_size(std::string &stype) {
    if (stype == "float32") return 4;
    if (stype == "int64") return 8;
    if (stype == "byte") return 1;
    if (stype == "uint1") return 1;
    if (stype == "int32") return 4;
    std::cout << stype << " is undefined" << std::endl;
    assert(false);
    return 0;
}

#define PARAM_MAGIC "TVM_MODEL_PARAMS"

ModelParam* ModelParamParser::parse_from_file(const char* param_file) {
    FILE* fp;
    fp = fopen(param_file, "rb"); 
    char magic[sizeof(PARAM_MAGIC)];
    std::cout << "fread" << __FILE__ << __LINE__ << std::endl;
    size_t res = fread(magic, sizeof(char), sizeof(PARAM_MAGIC), fp);
    assert(res == sizeof(PARAM_MAGIC));
    assert(std::string(magic) == PARAM_MAGIC);
    
    uint64_t params_size;
    std::cout << "fread" << __FILE__ << __LINE__ << std::endl;
    res = fread(&params_size, sizeof(uint64_t), 1, fp);
    assert(res == 1);
    assert(params_size != 0);

    ModelParam* params = new ModelParam(params_size);
    for (uint64_t i = 0; i < params_size; i++) {
        char key_buf[256];
        uint64_t key_len = 0;
        while(true) {
            char c;
            res = fread(&c, sizeof(char), 1, fp);
            assert(res == 1);
            key_buf[key_len] = c;
            key_len++;
            if (c == '\0') break;
        }
        std::string key(key_buf);
        uint64_t array_size;
        res = fread(&array_size, sizeof(uint64_t), 1, fp);
        assert(res == 1);
        assert(array_size != 0);
        std::vector<float> array(array_size);
        array.resize(array_size);
        res = fread(array.data(), sizeof(float), array_size, fp);
        assert(res == array_size);
        params->insert({key, array});
    }
    return params;
}