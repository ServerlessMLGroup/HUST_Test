import os

import json
import sys


############################################################################
#
#  This script is used to combine the two json files generated from TVM
#  to a final schedule json file which can be used by the dnn_rt_scheduler.
#
#  Usage: python3 generate-final-schedule.py reef/resnet18.cu reef/resnet18.schedule.json reef/resnet18.graph.json
#
#  The first json file(raw_schedule_file.json) is generated by TVM runtime,
#  which contains the basic kernel schedule and kernel parameters.
#
#  The second json file(graph_json.json) is generated by TVM backend(modified version),
#  which contains host function information and device stroage information.
#
############################################################################
def generate_final_schedule(source_code_lines, schedule_raw, graph):
    def split_function_declaration(line):
        parts = line.split("(")
        parameters_str = parts[1].split(")")[0]
        left_parts = parts[0].split(" ")
        name = left_parts[-1]
        return_type = left_parts[-2]
        header = " ".join(left_parts[:-2])
        parameter_str_list = parameters_str.split(", ")
        parameters = []
        for param_str in parameter_str_list:
            parts = param_str.split(" ")
            param_name = parts[-1]
            param_type = " ".join(parts[:-1])
            parameters.append({"name": param_name, "type": param_type})
        return header, return_type, name, parameters

    # 1. storage info from graph_json

    storage_id = graph["attrs"]["storage_id"][1]
    ## FIXME: a hack here
    ## to avoid buffer reuse, we replace storage_id to itself.
    for i in range(len(storage_id)):
        storage_id[i] = i

    storage = []
    for i in range(max(storage_id) + 1):
        storage.append({"name": "null", "size": 0, "stype": "null"})

    arg_idx = []

    for i in range(len(storage_id)):
        shape = graph["attrs"]["shape"][1][i]
        t = graph["attrs"]["dltype"][1][i]
        size = 1
        for j in shape:
            size = size * j
        sid = storage_id[i]
        if storage[sid]["size"] < size:
            storage[sid]["size"] = size
            storage[sid]["stype"] = t

    for a in graph["arg_nodes"]:
        sid = storage_id[a]
        name = graph["nodes"][a]["name"]
        storage[sid]["name"] = name
        arg_idx.append(sid)

    # 2. append dynamic allocated storage
    temp_storage_begin = len(storage)
    for temp_arg in schedule_raw["temp_args"]:
        storage.append({"name": "temp_arg", "size": temp_arg, "stype": "byte"})

    # 3. remap the kernel args
    i = 0
    kernels = []
    node_row_ptr = graph["node_row_ptr"]
    for j in range(len(graph["nodes"])):
        node = graph["nodes"][j]
        if node["op"] == "null":
            continue
        if node["attrs"]["func_name"] == "__nop":
            continue

        schedule_func = schedule_raw["funcs"][i]
        while len(schedule_func["kernels"]) == 0:
            i = i + 1
            schedule_func = schedule_raw["funcs"][i]

        if schedule_func["name"] != node["attrs"]["func_name"]:
            raise Exception("schedule name != node name, %s != %s" % (schedule_func["name"], node["name"]))
        # if node["attrs"]["num_outputs"] != "1":
        #     print(node["attrs"]["num_outputs"])
        #     raise Exception("node output != 1")
        host_inputs = []
        for inp in node["inputs"]:
            host_inputs.append(node_row_ptr[inp[0]] + inp[1])
        for idx in range(int(node["attrs"]["num_outputs"])):
            host_inputs.append(node_row_ptr[j] + idx)
        for kernel in schedule_func["kernels"]:
            new_args = []
            for arg in kernel["args"]:
                if arg < 0:
                    new_args.append(temp_storage_begin - arg - 1)
                else:
                    new_args.append(storage_id[host_inputs[arg]])
            kernels.append({"name": kernel["name"], "launch_params": kernel["launch_params"], "args": new_args})
        i = i + 1

    output_idx = graph["heads"][0][0]
    storage[storage_id[output_idx]]["name"] = "output"

    schedule = {
        "storage": storage,
        "kernels": kernels,
        "args": arg_idx
    }

    # 4. generate shared memory usage

    func_name = ""
    shared_memory = 0

    result = {}

    for line in source_code_lines:
        if line.find("void") != -1:
            # save old values
            if func_name != "":
                if shared_memory < 4:
                    shared_memory = 4
                result[func_name] = shared_memory

            _, _, curr_func_name, _ = split_function_declaration(line)
            func_name = curr_func_name
            shared_memory = 4
        if line.find("__shared__") != -1:
            # __shared__ float x[123];
            size = line.split("[")[1].split("]")[0]
            shared_memory = shared_memory + int(size) * 4

    if func_name != "":
        if shared_memory < 4:
            shared_memory = 4
        result[func_name] = shared_memory

    schedule["shared_memory"] = result
    return schedule


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Useage: source_code raw_scheduler_file graph_json_file ")
        exit(0)

    f = open(sys.argv[1], "r")
    source_code_lines = f.readlines()
    f.close()

    f = open(sys.argv[2], "r")
    schedule_raw = json.loads(f.read())
    f.close()

    f = open(sys.argv[3], "r")
    graph = json.loads(f.read())
    f.close()

    schedule = generate_final_schedule(source_code_lines, schedule_raw, graph)

    file_path = os.path.dirname(os.path.abspath(__file__)) + '/reef'
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    with open("reef/resnet18-final.cu", "w") as f:
        print(json.dumps(schedule, indent=4), file=f)
        f.close()