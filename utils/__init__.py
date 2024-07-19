from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from utils.preprocessing import clean_text_imdb
from utils.dataloader import load_train_test_imdb_data,load_mr_data
import numpy as np
import datasets
import matplotlib.pyplot as plt
import math
import pathlib
import glob
import xlsxwriter
import json
vectorizer = CountVectorizer(stop_words="english",
                                preprocessor=clean_text_imdb, 
                                min_df=0)

def survey_huggingface_dataset(name:str,input_name:str="text"):
    dataset = datasets.load_dataset(name)
    train_data = dataset["train"]
    test_data = dataset["test"]
    training_features = vectorizer.fit_transform(train_data[input_name])
    test_features = vectorizer.transform(test_data[input_name])
    num_element = 0
    num_element+= training_features.sum()
    num_element+= test_features.sum()
    average_length = num_element/(training_features.shape[0]+test_features.shape[0])
    
    return training_features.shape[0],test_features.shape[0],average_length


def plot_vocab_count(vectorizer,input_texts,name:str="sample.jpg",bins:int=100):
    
    plot_data = []
    vocab_count = [0 for i in range(len(vectorizer.vocabulary_))]
    indices = vectorizer.transform(input_texts)
    indices = indices.sum(axis=0)
    indices = np.squeeze(np.asarray(indices))
    vocab_count = indices.tolist()
    vocab_count = sorted(vocab_count,reverse=True)
    vocab_count = vocab_count[1:]
    sum_vocab = sum(vocab_count)
    print(sum_vocab)
    #vocab_count = [i/(sum_vocab) for i in vocab_count]
    bin_combine = len(vocab_count)//bins
    
    for i in range(bins):
        plot_data.append(((sum(vocab_count[i*bin_combine:(i+1)*bin_combine])/bin_combine)+1))
    plt.bar(range(0,bins), plot_data)
    plt.savefig("/home/ubuntu/Robustness_Gym/plot/"+name)
    plt.close()
    

def draw_excel(path_to_results):
    workbook = xlsxwriter.Workbook(path_to_results+"/results.xlsx")
    worksheet = workbook.add_worksheet()
    cellformat = workbook.add_format()
    cellformat.set_align('center')
    cellformat.set_align('vcenter')
    
    f = open(path_to_results+'/result.json')
    clean_acc = json.load(f)
    f.close()
    lst_attack = list(clean_acc.keys())
    first_collumn = 65
    worksheet.merge_range(f'{chr(first_collumn)}1:{chr(first_collumn)}2',"Models",cell_format=cellformat)
    worksheet.merge_range(f'{chr(first_collumn+1)}1:{chr(first_collumn+1)}2',"Clean Accuracy (%)",cell_format=cellformat)
    for i in range(0,len(lst_attack)):
        # Merge 3 cells.
        worksheet.merge_range(f'{chr(first_collumn+(i*2)+2)}1:{chr(first_collumn+(i*2)+3)}1',lst_attack[i],cell_format=cellformat)
        worksheet.write(f'{chr(first_collumn+(i*2)+2)}2', 'AuA(%) (ASR(%)↓)',cellformat)
        worksheet.write(f'{chr(first_collumn+(i*2)+3)}2', 'Avg. Query↓',cellformat)
        results = list(clean_acc[lst_attack[i]].keys())
        results = sorted(results, key=lambda item: (int(item.split("_")[-1])+len(item.split("_"))*(100)
                               if item.split("_")[-1].isdigit() else float(0), item))
        for k in range(len(results)):

            worksheet.write(f'{chr(first_collumn)}{str(k+3)}', results[k],cellformat)
            percentage  = clean_acc[lst_attack[i]][results[k]]["Attack success rate"]
            AuA = clean_acc[lst_attack[i]][results[k]]["Accuracy under attack"]
            worksheet.write(f'{chr(first_collumn+(i*2)+2)}{str(k+3)}', f"{AuA} ({percentage})",cellformat)
            querries  = clean_acc[lst_attack[i]][results[k]]["Avg num queries"]
            worksheet.write(f'{chr(first_collumn+(i*2)+3)}{str(k+3)}', f"{querries}",cellformat)
            worksheet.write(f'{chr(first_collumn+1)}{str(k+3)}', clean_acc[lst_attack[i]][results[k]]["Original accuracy"],cellformat)

    worksheet.autofit()
    workbook.close()
    pass
def p2f(x):
   return float(x.strip('%'))/100

def draw_excel2(path_to_results,name = "IMDB_clean_accuracy",iteration = 3):
    workbook = xlsxwriter.Workbook(f"{path_to_results}/{name}_results.xlsx")
    worksheet = workbook.add_worksheet()
    cellformat = workbook.add_format()
    cellformat.set_align('center')
    cellformat.set_align('vcenter')
    results = {}
    for i in range(iteration):
        f = open(f"{path_to_results}/{name}_{i}.json")
        clean_acc = json.load(f)
        for k in clean_acc.keys():
            if k not in results.keys():
                results[k] = p2f(clean_acc[k])/iteration
            else:
                results[k] += p2f(clean_acc[k])/iteration
    first_collumn = 65
    worksheet.write(f'{chr(first_collumn)}1', 'noise_type\intensity',cellformat)
    row = 2
    position = 1
    record_noise = {}
    record_pos = {}
    for i  in list(results.keys())[1:]:
        names = i.split("_")
        noise_level = names[-1]
        name = "_".join(names[4:-1])
        if name not in record_pos.keys():
            record_pos[name]=row
            worksheet.write(f'{chr(first_collumn)}{str(record_pos[name])}', name,cellformat)
            row+=1
        if noise_level not in record_noise.keys():
            record_noise[noise_level]=position+first_collumn
            worksheet.write(f'{chr(record_noise[noise_level])}1', noise_level,cellformat)
            position+=1
        worksheet.write(f'{chr(record_noise[noise_level])}{str(record_pos[name])}', f"{results[i]*100:.2f}%",cellformat)
    print(record_pos)
    print(record_noise)
    worksheet.autofit()
    workbook.close()
    pass

