# AdvFooler-Adversarial-Text-Defender
AdvFooler is a defender that fools the Textual Adversarial Attacker (The Fooler) via Randomizing Latent Representations

## Creating the environment.
To create the environment for the experiment, run this command.

`conda env create -f environment.yml`
## Running the experiment.

# Supported attacks and dataset.

This repository supports these attacks and datasets. These attack implementations are modified versions of [TextAttack](https://github.com/QData/TextAttack):
 - Adversarial attacks:
   - TextFooler
    - TextBugger
    - BERTAttack
    - HotFlip
    - HardLabel
 - Dataset:
    - IMDB
    - AGNEWS

# Running the experiments.
 2. Run using python commands:
   - Run the following command to perform the defense
   
```python main.py --load_path $load_path --attack_method $attack_method --def_position $def_position --noise_intensity $noise_intensity --parallel --num_workers_per_device $num```

   - you can adjust the number of workers for each GPU by adding `--num_workers_per_device` for the commands in the bash script.
   - you can also limit the GPUs by passing CUDA_VISIBLE_DEVICES=(GPU IDs) before the commands.
   - example: `python main.py --load_path textattack/bert-base-uncased-ag-news --attack_method textfooler --def_position post_att_cls --noise_intensity 0.8 --parallel --num_workers_per_device 2`
