import os
import json

def readjust_result(result_path,results_notattacked):
    results = os.listdir(result_path)
    for result in results_notattacked.keys() :
        f = open(f'{result_path}/{result}.json')
        curren_result = json.load(f)
        already_targetted = curren_result["Total Attacked Instances"] - results_notattacked[result]
        curren_result["Successful Instances"] =int( curren_result["Successful Instances"]-already_targetted)
        curren_result["Total Attacked Instances"] = int(results_notattacked[result])
        curren_result["Attack Success Rate"] = float(curren_result["Successful Instances"]/curren_result["Total Attacked Instances"])
        f.close()
        print(curren_result)
        with open(f'{result_path}/{result}_readjusted.json', 'w') as f:
            json.dump(curren_result, f)
    pass

def process_text_result(path):
    results = next(os.walk(path))[1]

    res_dict = {}
    list_models = []
    list_attacks = []
    for i in results:
        res_attack = {}
        attack_method = os.listdir(f"{path}/{i}")
        list_attacks.extend(attack_method)
        for k in attack_method:
            models_list = os.listdir(f"{path}/{i}/{k}")
            model_results = {}
            for j in models_list:
                banned_list = []
                skip = False
                print(j)
                for banned in banned_list:
                    if banned in j: 
                        skip = True
                        break
                if skip==True:
                    continue
                with open(f"{path}/{i}/{k}/{j}") as f:
                    res = f.readlines()
                    res_spec = {}
                    for spec in res[-9:]:
                        res_spec[spec.split(":")[0]] = spec.split(":")[1].replace("\n","")
                model_name = ".".join(j.split("-")[-1].split(".")[:-1])
                list_models.append(model_name)
                model_results[model_name] = res_spec
            res_attack[k] = model_results
        res_dict[i] = res_attack
    list_models = list(set(list_models))
    list_attacks = list(set(list_attacks))
    final_res = {}
    for i in list_attacks:
        model_attack_res = {}
        for j in list_models:
            all_model_res = []
            for k in results:
                if i in res_dict[k] and j in res_dict[k][i]:
                    all_model_res.append(res_dict[k][i][j])
            num = len(all_model_res)
            mold_res ={}
            print(j)
            print(all_model_res)
            for k in all_model_res[0].keys():
                total_value = 0
                for l in range(num):
                    if "%" in all_model_res[0][k]:
                        total_value+=float(all_model_res[l][k].split("%")[0])
                    else:
                        total_value+=float(all_model_res[l][k])
                mold_result = "{:.2f}".format(total_value/num)
                if "%" in all_model_res[0][k]:
                    mold_result+="%"
                mold_res[k] = mold_result
            model_attack_res[j] = mold_res
        final_res[i]=model_attack_res
    json_object = json.dumps(final_res, indent=4)
 
    # Writing to sample.json
    with open(f"{path}/result.json", "w") as outfile:
        outfile.write(json_object)
    pass

if __name__ == '__main__':
    process_text_result("noise_defense_attack_result/paper_default setting/AGNEWS")