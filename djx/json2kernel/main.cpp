#include "model.h"
#include "log.h"
#include <bits/unique_ptr.h>

int main() {
    log("enter main func");
    std::unique_ptr<Model> model;
    log("preate unique_ptr");
    model.reset(Model::from_json("/home/husterdjx/research/HUST_Test/djx/json2kernel/resnet18-final.json"));
    model.reset();
    return 0;
}