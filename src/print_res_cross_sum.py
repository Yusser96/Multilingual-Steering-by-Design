import json 
import os
import glob
from collections import defaultdict
import pandas as pd
import sys

model = sys.argv[1]
alpha = sys.argv[2]
# task = sys.argv[3]

# alpha = "5.0"
# model = "meta-llama/Llama-3.1-8B"

# alpha = "100.0"
# model = "google/gemma-2-9b"

task ="cross_sum" #"xnli" #"belebele"

path = f"cross_sum-all_langs2-saes/{alpha}/{model}"



def avg_res_all():
    final_res = defaultdict(list)
    base_acc = 0
    base_bleu = 0
    base_comet = 0
    base_cnt = 0
    base_flag = True


    for layer in os.listdir(path):
        
        layer_res_path = os.path.join(path,layer)
        tmp_saes = os.listdir(layer_res_path)
        saes = []
        for s in tmp_saes:
            if "Yusser" in s:
                saes.extend(os.listdir(os.path.join(layer_res_path, "Yusser")))
            else:
                saes.append(s)


        # json_files = glob.glob(os.path.join(layer_res_path, f"*/{task}/*.json"))
        json_files = glob.glob(os.path.join(layer_res_path, "**",f"{task}", "*.json"), recursive=True)

        # print(layer, saes,len(json_files))


        final_res["layer"].append(int(layer))

        for s in saes:
            acc = 0
            bleu = 0
            comet = 0
            cnt = 0
            print(s)

            for j in json_files:
                if s in j:
                    print(j)
                    with open(j, "r", encoding="utf-8") as f:
                        results = json.load(f)
                        # print(results["base_accuray"],results["accuray"]) 
                        j_name = j.split("/")[-1].replace(".json","").split("-") 
                        source_lang = j_name[0]
                        tgt_lang = j_name[1]

                        acc += results["mod"]["langid_acc"] * 100
                        cnt += 1

                        bleu += results["mod"]["rougeL"] * 100

                        comet += results["mod"]["avg_lase_success"]

                        if base_flag:
                            base_acc += results["base"]["langid_acc"] * 100
                            base_bleu += results["base"]["rougeL"] * 100
                            base_comet += results["base"]["avg_lase_success"]
                            base_cnt += 1

            base_flag = False
            
            final_res[s].append(f"{round(acc/cnt,2)}/{round(bleu/cnt,2)}/{round(comet/cnt,2)}")
        
        final_res["prompt"].append(f"{round(base_acc/base_cnt,2)}/{round(base_bleu/base_cnt,2)}/{round(base_comet/base_cnt,2)}")

        # final_res["layer"].append("prompt")



    df = pd.DataFrame(final_res)
    df = df.sort_values('layer')

    out_path = f"latex2/{task}"
    os.makedirs(out_path,exist_ok=True)

    df.to_csv(os.path.join(out_path, f"{model.split('/')[-1]}_all.tsv"),sep="\t",index=False)


def avg_res_diff():
    final_res = defaultdict(list)
    base_acc = 0
    base_bleu = 0
    base_comet = 0
    base_cnt = 0
    base_flag = True


    for layer in os.listdir(path):
        
        layer_res_path = os.path.join(path,layer)
        tmp_saes = os.listdir(layer_res_path)
        saes = []
        for s in tmp_saes:
            if "Yusser" in s:
                saes.extend(os.listdir(os.path.join(layer_res_path, "Yusser")))
            else:
                saes.append(s)


        # json_files = glob.glob(os.path.join(layer_res_path, f"*/{task}/*.json"))
        json_files = glob.glob(os.path.join(layer_res_path, "**",f"{task}", "*.json"), recursive=True)

        # print(layer, saes,len(json_files))


        final_res["layer"].append(int(layer))

        for s in saes:
            acc = 0
            bleu = 0
            comet = 0
            cnt = 0
            print(s)

            for j in json_files:
                if s in j:
                    print(j)
                    with open(j, "r", encoding="utf-8") as f:
                        results = json.load(f)
                        # print(results["base_accuray"],results["accuray"]) 
                        j_name = j.split("/")[-1].replace(".json","").split("-") 
                        source_lang = j_name[0]
                        tgt_lang = j_name[1]
                        if source_lang == tgt_lang:
                            continue

                        acc += results["mod"]["langid_acc"] * 100
                        cnt += 1

                        bleu += results["mod"]["rougeL"] * 100

                        comet += results["mod"]["avg_lase_success"]

                        if base_flag:
                            base_acc += results["base"]["langid_acc"] * 100
                            base_bleu += results["base"]["rougeL"] * 100
                            base_comet += results["base"]["avg_lase_success"]
                            base_cnt += 1

            base_flag = False
            
            final_res[s].append(f"{round(acc/cnt,2)}/{round(bleu/cnt,2)}/{round(comet/cnt,2)}")
        
        final_res["prompt"].append(f"{round(base_acc/base_cnt,2)}/{round(base_bleu/base_cnt,2)}/{round(base_comet/base_cnt,2)}")

        # final_res["layer"].append("prompt")



    df = pd.DataFrame(final_res)
    df = df.sort_values('layer')

    out_path = f"latex2/{task}"
    os.makedirs(out_path,exist_ok=True)

    df.to_csv(os.path.join(out_path, f"{model.split('/')[-1]}_diff.tsv"),sep="\t",index=False)


def avg_res_same():
    final_res = defaultdict(list)
    base_acc = 0
    base_bleu = 0
    base_comet = 0
    base_cnt = 0
    base_flag = True


    for layer in os.listdir(path):
        
        layer_res_path = os.path.join(path,layer)
        tmp_saes = os.listdir(layer_res_path)
        saes = []
        for s in tmp_saes:
            if "Yusser" in s:
                saes.extend(os.listdir(os.path.join(layer_res_path, "Yusser")))
            else:
                saes.append(s)


        # json_files = glob.glob(os.path.join(layer_res_path, f"*/{task}/*.json"))
        json_files = glob.glob(os.path.join(layer_res_path, "**",f"{task}", "*.json"), recursive=True)

        # print(layer, saes,len(json_files))


        final_res["layer"].append(int(layer))

        for s in saes:
            acc = 0
            bleu = 0
            comet = 0
            cnt = 0
            print(s)

            for j in json_files:
                if s in j:
                    print(j)
                    with open(j, "r", encoding="utf-8") as f:
                        results = json.load(f)
                        # print(results["base_accuray"],results["accuray"]) 
                        j_name = j.split("/")[-1].replace(".json","").split("-") 
                        source_lang = j_name[0]
                        tgt_lang = j_name[1]
                        if source_lang != tgt_lang:
                            continue

                        acc += results["mod"]["langid_acc"] * 100
                        cnt += 1

                        bleu += results["mod"]["rougeL"] * 100

                        comet += results["mod"]["avg_lase_success"]

                        if base_flag:
                            base_acc += results["base"]["langid_acc"] * 100
                            base_bleu += results["base"]["rougeL"] * 100
                            base_comet += results["base"]["avg_lase_success"]
                            base_cnt += 1

            base_flag = False
            
            final_res[s].append(f"{round(acc/cnt,2)}/{round(bleu/cnt,2)}/{round(comet/cnt,2)}")
        
        final_res["prompt"].append(f"{round(base_acc/base_cnt,2)}/{round(base_bleu/base_cnt,2)}/{round(base_comet/base_cnt,2)}")

        # final_res["layer"].append("prompt")



    df = pd.DataFrame(final_res)
    df = df.sort_values('layer')

    out_path = f"latex2/{task}"
    os.makedirs(out_path,exist_ok=True)

    df.to_csv(os.path.join(out_path, f"{model.split('/')[-1]}_same.tsv"),sep="\t",index=False)



def avg_res_per_lang_all():
    languages = ["es", "ru", "ar", "hi", "tr"]
    for lang in languages: 
        final_res = defaultdict(list)
        base_acc = 0
        base_bleu = 0
        base_comet = 0
        base_cnt = 0
        base_flag = True


        for layer in os.listdir(path):
            
            layer_res_path = os.path.join(path,layer)
            tmp_saes = os.listdir(layer_res_path)
            saes = []
            for s in tmp_saes:
                if "Yusser" in s:
                    saes.extend(os.listdir(os.path.join(layer_res_path, "Yusser")))
                else:
                    saes.append(s)


            # json_files = glob.glob(os.path.join(layer_res_path, f"*/{task}/*.json"))
            # json_files = glob.glob(os.path.join(layer_res_path, "**",f"{task}", "*.json"), recursive=True)
            json_files = glob.glob(os.path.join(layer_res_path, "**",f"{task}", f"{lang}-*.json"), recursive=True)

            # print(layer, saes,len(json_files))


            final_res["layer"].append(int(layer))

            for s in saes:
                acc = 0
                bleu = 0
                comet = 0
                cnt = 0
                print(s)

                for j in json_files:
                    if s in j:
                        print(j)
                        with open(j, "r", encoding="utf-8") as f:
                            results = json.load(f)
                            # print(results["base_accuray"],results["accuray"]) 
                            j_name = j.split("/")[-1].replace(".json","").split("-") 
                            source_lang = j_name[0]
                            tgt_lang = j_name[1]


                            acc += results["mod"]["langid_acc"] * 100
                            cnt += 1

                            bleu += results["mod"]["rougeL"] * 100

                            comet += results["mod"]["avg_lase_success"]

                            if base_flag:
                                base_acc += results["base"]["langid_acc"] * 100
                                base_bleu += results["base"]["rougeL"] * 100
                                base_comet += results["base"]["avg_lase_success"]
                                base_cnt += 1

                base_flag = False
                
                final_res[s].append(f"{round(acc/cnt,2)}/{round(bleu/cnt,2)}/{round(comet/cnt,2)}")
            
            final_res["prompt"].append(f"{round(base_acc/base_cnt,2)}/{round(base_bleu/base_cnt,2)}/{round(base_comet/base_cnt,2)}")

            # final_res["layer"].append("prompt")



        df = pd.DataFrame(final_res)
        df = df.sort_values('layer')

        out_path = f"latex2/{task}/per_lang"
        os.makedirs(out_path,exist_ok=True)

        df.to_csv(os.path.join(out_path, f"{model.split('/')[-1]}_all.{lang}.tsv"),sep="\t",index=False)

def avg_res_per_lang_same():
    languages = ["es", "ru", "ar", "hi", "tr"]
    for lang in languages: 
        final_res = defaultdict(list)
        base_acc = 0
        base_bleu = 0
        base_comet = 0
        base_cnt = 0
        base_flag = True


        for layer in os.listdir(path):
            
            layer_res_path = os.path.join(path,layer)
            tmp_saes = os.listdir(layer_res_path)
            saes = []
            for s in tmp_saes:
                if "Yusser" in s:
                    saes.extend(os.listdir(os.path.join(layer_res_path, "Yusser")))
                else:
                    saes.append(s)


            # json_files = glob.glob(os.path.join(layer_res_path, f"*/{task}/*.json"))
            # json_files = glob.glob(os.path.join(layer_res_path, "**",f"{task}", "*.json"), recursive=True)
            json_files = glob.glob(os.path.join(layer_res_path, "**",f"{task}", f"{lang}-*.json"), recursive=True)

            # print(layer, saes,len(json_files))


            final_res["layer"].append(int(layer))

            for s in saes:
                acc = 0
                bleu = 0
                comet = 0
                cnt = 0
                print(s)

                for j in json_files:
                    if s in j:
                        print(j)
                        with open(j, "r", encoding="utf-8") as f:
                            results = json.load(f)
                            # print(results["base_accuray"],results["accuray"]) 
                            j_name = j.split("/")[-1].replace(".json","").split("-") 
                            source_lang = j_name[0]
                            tgt_lang = j_name[1]
                            if source_lang != tgt_lang:
                                continue

                            acc += results["mod"]["langid_acc"] * 100
                            cnt += 1

                            bleu += results["mod"]["rougeL"] * 100

                            comet += results["mod"]["avg_lase_success"]

                            if base_flag:
                                base_acc += results["base"]["langid_acc"] * 100
                                base_bleu += results["base"]["rougeL"] * 100
                                base_comet += results["base"]["avg_lase_success"]
                                base_cnt += 1

                base_flag = False
                
                final_res[s].append(f"{round(acc/cnt,2)}/{round(bleu/cnt,2)}/{round(comet/cnt,2)}")
            
            final_res["prompt"].append(f"{round(base_acc/base_cnt,2)}/{round(base_bleu/base_cnt,2)}/{round(base_comet/base_cnt,2)}")

            # final_res["layer"].append("prompt")



        df = pd.DataFrame(final_res)
        df = df.sort_values('layer')

        out_path = f"latex2/{task}/per_lang"
        os.makedirs(out_path,exist_ok=True)

        df.to_csv(os.path.join(out_path, f"{model.split('/')[-1]}_same.{lang}.tsv"),sep="\t",index=False)




def avg_res_per_lang_diff():
    languages = ["es", "ru", "ar", "hi", "tr"]
    for lang in languages: 
        final_res = defaultdict(list)
        base_acc = 0
        base_bleu = 0
        base_comet = 0
        base_cnt = 0
        base_flag = True


        for layer in os.listdir(path):
            
            layer_res_path = os.path.join(path,layer)
            tmp_saes = os.listdir(layer_res_path)
            saes = []
            for s in tmp_saes:
                if "Yusser" in s:
                    saes.extend(os.listdir(os.path.join(layer_res_path, "Yusser")))
                else:
                    saes.append(s)


            # json_files = glob.glob(os.path.join(layer_res_path, f"*/{task}/*.json"))
            # json_files = glob.glob(os.path.join(layer_res_path, "**",f"{task}", "*.json"), recursive=True)
            json_files = glob.glob(os.path.join(layer_res_path, "**",f"{task}", f"{lang}-*.json"), recursive=True)

            # print(layer, saes,len(json_files))


            final_res["layer"].append(int(layer))

            for s in saes:
                acc = 0
                bleu = 0
                comet = 0
                cnt = 0
                print(s)

                for j in json_files:
                    if s in j:
                        print(j)
                        with open(j, "r", encoding="utf-8") as f:
                            results = json.load(f)
                            # print(results["base_accuray"],results["accuray"]) 
                            j_name = j.split("/")[-1].replace(".json","").split("-") 
                            source_lang = j_name[0]
                            tgt_lang = j_name[1]
                            if source_lang == tgt_lang:
                                continue

                            acc += results["mod"]["langid_acc"] * 100
                            cnt += 1

                            bleu += results["mod"]["rougeL"] * 100

                            comet += results["mod"]["avg_lase_success"]

                            if base_flag:
                                base_acc += results["base"]["langid_acc"] * 100
                                base_bleu += results["base"]["rougeL"] * 100
                                base_comet += results["base"]["avg_lase_success"]
                                base_cnt += 1

                base_flag = False
                
                final_res[s].append(f"{round(acc/cnt,2)}/{round(bleu/cnt,2)}/{round(comet/cnt,2)}")
            
            final_res["prompt"].append(f"{round(base_acc/base_cnt,2)}/{round(base_bleu/base_cnt,2)}/{round(base_comet/base_cnt,2)}")

            # final_res["layer"].append("prompt")



        df = pd.DataFrame(final_res)
        df = df.sort_values('layer')

        out_path = f"latex2/{task}/per_lang"
        os.makedirs(out_path,exist_ok=True)

        df.to_csv(os.path.join(out_path, f"{model.split('/')[-1]}_diff.{lang}.tsv"),sep="\t",index=False)





def avg_res_per_tgt_lang_all():
    languages = ["es", "ru", "ar", "hi", "tr"]
    for lang in languages: 
        final_res = defaultdict(list)
        base_acc = 0
        base_bleu = 0
        base_comet = 0
        base_cnt = 0
        base_flag = True


        for layer in os.listdir(path):
            
            layer_res_path = os.path.join(path,layer)
            tmp_saes = os.listdir(layer_res_path)
            saes = []
            for s in tmp_saes:
                if "Yusser" in s:
                    saes.extend(os.listdir(os.path.join(layer_res_path, "Yusser")))
                else:
                    saes.append(s)


            # json_files = glob.glob(os.path.join(layer_res_path, f"*/{task}/*.json"))
            # json_files = glob.glob(os.path.join(layer_res_path, "**",f"{task}", "*.json"), recursive=True)
            json_files = glob.glob(os.path.join(layer_res_path, "**",f"{task}", f"*-{lang}.json"), recursive=True)

            # print(layer, saes,len(json_files))


            final_res["layer"].append(int(layer))

            for s in saes:
                acc = 0
                bleu = 0
                comet = 0
                cnt = 0
                print(s)

                for j in json_files:
                    if s in j:
                        print(j)
                        with open(j, "r", encoding="utf-8") as f:
                            results = json.load(f)
                            # print(results["base_accuray"],results["accuray"]) 
                            j_name = j.split("/")[-1].replace(".json","").split("-") 
                            source_lang = j_name[0]
                            tgt_lang = j_name[1]

                            acc += results["mod"]["langid_acc"] * 100
                            cnt += 1

                            bleu += results["mod"]["rougeL"] * 100

                            comet += results["mod"]["avg_lase_success"]

                            if base_flag:
                                base_acc += results["base"]["langid_acc"] * 100
                                base_bleu += results["base"]["rougeL"] * 100
                                base_comet += results["base"]["avg_lase_success"]
                                base_cnt += 1

                base_flag = False
                
                final_res[s].append(f"{round(acc/cnt,2)}/{round(bleu/cnt,2)}/{round(comet/cnt,2)}")
            
            final_res["prompt"].append(f"{round(base_acc/base_cnt,2)}/{round(base_bleu/base_cnt,2)}/{round(base_comet/base_cnt,2)}")

            # final_res["layer"].append("prompt")



        df = pd.DataFrame(final_res)
        df = df.sort_values('layer')

        out_path = f"latex2/{task}/per_tgt_lang"
        os.makedirs(out_path,exist_ok=True)

        df.to_csv(os.path.join(out_path, f"{model.split('/')[-1]}_all.{lang}.tsv"),sep="\t",index=False)



def avg_res_per_tgt_lang_same():
    languages = ["es", "ru", "ar", "hi", "tr"]
    for lang in languages: 
        final_res = defaultdict(list)
        base_acc = 0
        base_bleu = 0
        base_comet = 0
        base_cnt = 0
        base_flag = True


        for layer in os.listdir(path):
            
            layer_res_path = os.path.join(path,layer)
            tmp_saes = os.listdir(layer_res_path)
            saes = []
            for s in tmp_saes:
                if "Yusser" in s:
                    saes.extend(os.listdir(os.path.join(layer_res_path, "Yusser")))
                else:
                    saes.append(s)


            # json_files = glob.glob(os.path.join(layer_res_path, f"*/{task}/*.json"))
            # json_files = glob.glob(os.path.join(layer_res_path, "**",f"{task}", "*.json"), recursive=True)
            json_files = glob.glob(os.path.join(layer_res_path, "**",f"{task}", f"*-{lang}.json"), recursive=True)

            # print(layer, saes,len(json_files))


            final_res["layer"].append(int(layer))

            for s in saes:
                acc = 0
                bleu = 0
                comet = 0
                cnt = 0
                print(s)

                for j in json_files:
                    if s in j:
                        print(j)
                        with open(j, "r", encoding="utf-8") as f:
                            results = json.load(f)
                            # print(results["base_accuray"],results["accuray"]) 
                            j_name = j.split("/")[-1].replace(".json","").split("-") 
                            source_lang = j_name[0]
                            tgt_lang = j_name[1]
                            if source_lang != tgt_lang:
                                continue

                            acc += results["mod"]["langid_acc"] * 100
                            cnt += 1

                            bleu += results["mod"]["rougeL"] * 100

                            comet += results["mod"]["avg_lase_success"]

                            if base_flag:
                                base_acc += results["base"]["langid_acc"] * 100
                                base_bleu += results["base"]["rougeL"] * 100
                                base_comet += results["base"]["avg_lase_success"]
                                base_cnt += 1

                base_flag = False
                
                final_res[s].append(f"{round(acc/cnt,2)}/{round(bleu/cnt,2)}/{round(comet/cnt,2)}")
            
            final_res["prompt"].append(f"{round(base_acc/base_cnt,2)}/{round(base_bleu/base_cnt,2)}/{round(base_comet/base_cnt,2)}")

            # final_res["layer"].append("prompt")



        df = pd.DataFrame(final_res)
        df = df.sort_values('layer')

        out_path = f"latex2/{task}/per_tgt_lang"
        os.makedirs(out_path,exist_ok=True)

        df.to_csv(os.path.join(out_path, f"{model.split('/')[-1]}_same.{lang}.tsv"),sep="\t",index=False)




def avg_res_per_tgt_lang_diff():
    languages = ["es", "ru", "ar", "hi", "tr"]
    for lang in languages: 
        final_res = defaultdict(list)
        base_acc = 0
        base_bleu = 0
        base_comet = 0
        base_cnt = 0
        base_flag = True


        for layer in os.listdir(path):
            
            layer_res_path = os.path.join(path,layer)
            tmp_saes = os.listdir(layer_res_path)
            saes = []
            for s in tmp_saes:
                if "Yusser" in s:
                    saes.extend(os.listdir(os.path.join(layer_res_path, "Yusser")))
                else:
                    saes.append(s)


            # json_files = glob.glob(os.path.join(layer_res_path, f"*/{task}/*.json"))
            # json_files = glob.glob(os.path.join(layer_res_path, "**",f"{task}", "*.json"), recursive=True)
            json_files = glob.glob(os.path.join(layer_res_path, "**",f"{task}", f"*-{lang}*.json"), recursive=True)

            # print(layer, saes,len(json_files))


            final_res["layer"].append(int(layer))

            for s in saes:
                acc = 0
                bleu = 0
                comet = 0
                cnt = 0
                print(s)

                for j in json_files:
                    if s in j:
                        print(j)
                        with open(j, "r", encoding="utf-8") as f:
                            results = json.load(f)
                            # print(results["base_accuray"],results["accuray"]) 
                            j_name = j.split("/")[-1].replace(".json","").split("-") 
                            source_lang = j_name[0]
                            tgt_lang = j_name[1]
                            if source_lang == tgt_lang:
                                continue

                            acc += results["mod"]["langid_acc"] * 100
                            cnt += 1

                            bleu += results["mod"]["rougeL"] * 100

                            comet += results["mod"]["avg_lase_success"]

                            if base_flag:
                                base_acc += results["base"]["langid_acc"] * 100
                                base_bleu += results["base"]["rougeL"] * 100
                                base_comet += results["base"]["avg_lase_success"]
                                base_cnt += 1

                base_flag = False
                
                final_res[s].append(f"{round(acc/cnt,2)}/{round(bleu/cnt,2)}/{round(comet/cnt,2)}")
            
            final_res["prompt"].append(f"{round(base_acc/base_cnt,2)}/{round(base_bleu/base_cnt,2)}/{round(base_comet/base_cnt,2)}")

            # final_res["layer"].append("prompt")



        df = pd.DataFrame(final_res)
        df = df.sort_values('layer')

        out_path = f"latex2/{task}/per_tgt_lang"
        os.makedirs(out_path,exist_ok=True)

        df.to_csv(os.path.join(out_path, f"{model.split('/')[-1]}_diff.{lang}.tsv"),sep="\t",index=False)


avg_res_all()

avg_res_diff()
avg_res_same()

avg_res_per_lang_all()

avg_res_per_lang_diff()
avg_res_per_lang_same()

avg_res_per_tgt_lang_all()

avg_res_per_tgt_lang_diff()
avg_res_per_tgt_lang_same()