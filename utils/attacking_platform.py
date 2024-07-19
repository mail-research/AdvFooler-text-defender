import datasets
import OpenAttack as oa
import pathlib
import json
def attack_platform(dataset_name,test_set_name,victim,victim_name,attacker,attacker_name,dataset_mapping,num_workers=0):
    """
    It takes in a dataset name, a test set name, a victim model, a victim model name, an attacker, an
    attacker name, a dataset mapping, and a number of workers. 
    
    It then loads the dataset, creates an attack eval object, and evaluates the attack on the dataset. 
    
    It then creates a directory for the results, and saves the results in a json file.
    
    :param dataset_name: the name of the dataset you want to attack
    :param test_set_name: the name of the test set you want to attack
    :param victim: the model you want to attack
    :param victim_name: the name of the model you want to attack
    :param attacker: The attack to use
    :param attacker_name: the name of the attack you want to run
    :param dataset_mapping: This is a function that takes in a dataset and returns a dataset. This is
    useful if you want to do things like normalize the dataset, or add a label smoothing term
    :param num_workers: number of workers to use for the attack
    """
    
    dataset = datasets.load_dataset(dataset_name, split=test_set_name).map(function=dataset_mapping)
    attack_eval = oa.AttackEval(attacker, victim,metrics=[oa.metric.ModificationRate(),oa.metric.EditDistance()])
    result = attack_eval.eval(dataset, progress_bar=True,num_workers=num_workers)
    pathlib.Path(f'/home/ubuntu/Robustness_Gym/results/{dataset_name}/{attacker_name}').mkdir(parents=True, exist_ok=True) 

    with open(f"/home/ubuntu/Robustness_Gym/results/{dataset_name}/{attacker_name}/{victim_name}.json", "w") as outfile:
        json.dump(result, outfile)