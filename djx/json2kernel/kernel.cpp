#include <cstddef>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <assert.h>
#include "json.h"

class StorageInfo {
public:
    std::string name;
    size_t size;
    std::string stype;
};

class KernelInfo {
public:
    std::string name;
    uint32_t launch_params[6];
    std::vector<size_t> args;
};

class Model {
public:
    std::vector<StorageInfo> storage;
    std::vector<KernelInfo> kernels;
    std::vector<uint32_t> args;
    std::unordered_map<std::string, size_t> shared_memory;

public:
    static Model* from_json(const char* json_file);
    static size_t get_stype_size(std::string &stype);
};

Model* Model::from_json(const char* json_file) {
    std::ifstream fs(json_file);
    std::string tmp, str = "";

    while (getline(fs, tmp)) str += tmp;
    fs.close();

    JsonObject* jobj = JsonParser::parse(str);

    Model* m = new Model;
    for (auto sinfo : jobj->mval["storage"]->lval) {
        m->storage.push_back(StorageInfo{
            sinfo->mval["name"]->sval,
            sinfo->mval["size"]->ival,
            sinfo->mval["stype"]->sval
        });
    }

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

    for (auto arg : jobj->mval["args"]->lval) {
        m->args.push_back(arg->ival);
    }

    for (auto shared_memory : jobj->mval["shared_memory"]->mval) {
        m->shared_memory[shared_memory.first] = shared_memory.second->ival;
    }
    delete jobj;

    return m;
}