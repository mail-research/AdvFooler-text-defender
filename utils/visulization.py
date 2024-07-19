from IPython.core.display import display, HTML
import numpy as np
import pandas as pd

def html_render(x_orig, x_adv):
    x_orig_words = x_orig.split(' ')
    x_adv_words = x_adv.split(' ')
    orig_html = []
    adv_html = []
    # For now, we assume both original and adversarial text have equal lengths.
    assert(len(x_orig_words) == len(x_adv_words))
    for i in range(len(x_orig_words)):
        if x_orig_words[i] == x_adv_words[i]:
            orig_html.append(x_orig_words[i])
            adv_html.append(x_adv_words[i])
        else:
            orig_html.append(format("<b style='color:green'>%s</b>" %x_orig_words[i]))
            adv_html.append(format("<b style='color:red'>%s</b>" %x_adv_words[i]))
    
    orig_html = ' '.join(orig_html)
    adv_html = ' '.join(adv_html)
    return orig_html, adv_html

def visualize_csv(csv_path,output_path):

    df = pd.read_csv(csv_path)
    orig = df["orig"]
    augment = df["augmented"]
    output_res = ""
    for i in range(len(orig)):
        set_visuals = html_render(orig[i],augment[i])
        output_res += " <br>".join(set_visuals)+"<br><br>"
    with open(output_path,"w") as f:
        print(output_path)
        f.writelines(output_res)
        f.close()
    pass

if __name__ == "__main__":
    visualize_csv("/home/ubuntu/Robustness_Gym/out.csv","/home/ubuntu/Robustness_Gym/visual_mnb.html")