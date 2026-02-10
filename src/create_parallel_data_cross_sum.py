from datasets import load_dataset 


# languages = [   
#         "bo", "mt", "it", "es", "de", "ja", "ar", "zh", "af", "nl", 
#         "fr", "pt", "ru", "ko", "hi", "tr", "pl", "sv", "da", "no", "en"
#     ]

languages = {
    "en": "english",
    "es": "spanish",
    "ja": "japanese",
    "ar": "arabic",
    "zh": "chinese_simplified",
    "fr": "french",
    "pt": "portuguese",
    "ru": "russian",
    "ko": "korean",
    "hi": "hindi",
    "tr": "turkish",
}

dataset_name = "csebuetnlp/CrossSum"



from collections import defaultdict 
import hashlib 
def make_id(text): 
    return hashlib.md5(text.strip().lower().encode()).hexdigest() 


langs_by_id = defaultdict(set) 
sentences = defaultdict(dict) 

for l1 in languages:
    for l2 in languages: 
        if l1 != l2 and "en" in f"{l1}-{l2}": 
            try: 
                dataset = load_dataset("csebuetnlp/CrossSum",f"{languages[l1]}-{languages[l2]}",trust_remote_code=True)
                print(l1,l2) 
                #print(dataset["train"][0]) 
                cnt = 1
                for split in ["train", "test", "validation"]:
                    for item in dataset[split]: #[:10_000_000]: # or "validation"/"test" 
                        if cnt % 5_000_000 == 0:
                            break
                        cnt += 1
                        sid = make_id(item["source_url"]) 
                        langs_by_id[sid].add(l1) 
                        langs_by_id[sid].add(l2) 

                        sentences[sid][l1] = item
                        sentences[sid][l2] = item
            except Exception as e: 
                print("error:",f"{languages[l1]}-{languages[l2]}")
                # print(e)
                pass



from itertools import combinations

def count_sentences_by_lang_subsets(langs_by_id, target_langs, min_size=1):
    """
    For each subset of target_langs (size >= min_size),
    compute how many sentence IDs support all languages in that subset,
    sort results from highest to lowest count, and print them.
    """
    target_langs = set(target_langs)
    results = []

    for k in range(min_size, len(target_langs) + 1):
        for subset in combinations(sorted(target_langs), k):
            subset = frozenset(subset)
            count = sum(
                1 for langs in langs_by_id.values()
                if subset.issubset(langs)
            )
            results.append((subset, count))

    # Sort by count descending
    results.sort(key=lambda x: x[1], reverse=True)

    # Print

    for subset, count in results[:50]:
        print(f"{set(subset)}: {count} sentences")

    return set(results[0][0])

target_langs = set(languages)


new_target_langs = count_sentences_by_lang_subsets(
    langs_by_id,
    target_langs,
    min_size=6   # e.g., only bilingual or higher
)


fully_parallel = [
    sid for sid, langs in langs_by_id.items()
    if new_target_langs.issubset(langs)
]

print(len(sentences))

print(f"Found {len(fully_parallel)} sentences aligned across {new_target_langs}")

new_data = defaultdict(list) 

for sid in sentences:
    if sid in fully_parallel:
        for lang in new_target_langs:
            new_data[lang].append(sentences[sid][lang])

import json 
with open(f"data/{dataset_name.split('/')[-1]}-parallel.json", "w" , encoding="utf8") as f:
    json.dump(new_data,f, indent=4, ensure_ascii=False)
