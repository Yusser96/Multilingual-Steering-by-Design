
from math import exp
from typing import Dict, List, Tuple
from typing import List, Optional
# from jaxtyping import Float, Int
import torch
from tqdm import tqdm
import json


_model_output = "text"
_max_new_tokens = 64



import re


def init(device):
    import fasttext


    try:
        langid_model = fasttext.load_model('lid218e.bin')
    except:
        import urllib.request
        urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/nllb/lid/lid218e.bin', 'lid218e.bin')
        langid_model = fasttext.load_model('lid218e.bin')


    from evaluate import load as load_metric

    rouge = load_metric("rouge")


    return langid_model, rouge

def init_lase():
    from LaSE import LaSEScorer 
    lase_scorer = LaSEScorer()

    return lase_scorer

def compute_rouge_score(rouge, references, generated_texts):

    # results = rouge.compute(predictions=generated_texts, references=[[r] for r in references])
    results = rouge.compute(predictions=generated_texts, references=references)

    return results


def detect_language(langid_model, text):
    """Detect language using fasttext, return language code"""
    if not text.strip():
        return "unknown"
    
    clean_text = text.replace('\n', ' ').strip()
    if not clean_text:
        return "unknown"
    
    predictions = langid_model.predict(clean_text, k=1)
    flores_code = predictions[0][0].replace('__label__', '')
    
    # Convert FLORES code back to original simple code
    return flores_code


def compute_lase_score(lase_scorer, list_of_references,list_of_predictions, ref_lang):
    


    scores = lase_scorer.batched_score(
        list_of_references,
        list_of_predictions,
        target_lang=ref_lang,
        batch_size=32
    )

    results = [o.LaSE * 100 for o in scores]
    score = sum(results) / len(results)
    return score


def lang_to_code(lang_code: str) -> str:
    mapping = {
        "bo": "bod_Tibt",  # Tibetan
        "mt": "mlt_Latn",  # Maltese
        "it": "ita_Latn",  # Italian
        "es": "spa_Latn",  # Spanish
        "de": "deu_Latn",  # German
        "ja": "jpn_Jpan",  # Japanese
        "ar": "arb_Arab",  # Modern Standard Arabic
        "zh": "zho_Hans",  # Chinese (Simplified)
        "af": "afr_Latn",  # Afrikaans
        "nl": "nld_Latn",  # Dutch
        "fr": "fra_Latn",  # French
        "pt": "por_Latn",  # Portuguese
        "ru": "rus_Cyrl",  # Russian
        "ko": "kor_Hang",  # Korean
        "hi": "hin_Deva",  # Hindi
        "tr": "tur_Latn",  # Turkish
        "pl": "pol_Latn",  # Polish
        "sv": "swe_Latn",  # Swedish
        "da": "dan_Latn",  # Danish
        "no": "nob_Latn",   # Norwegian Bokmål
        "en": "eng_Latn"
    }
    return mapping.get(lang_code, "United States")  # default fallback

def get_language_prompts(target_lang):
    """Target-language prompts equivalent to English 'Summarize this article:'"""
    
    prompts = {
        "eng_Latn": "Summarize this article:",
        "bod_Tibt": "རྩོམ་ཡིག་འདི་མདོར་བསྡུས་བྱོས།",
        "mlt_Latn": "Agħti sommarju ta’ dan l-artiklu:",
        "ita_Latn": "Riassumi questo articolo:",
        "spa_Latn": "Resume este artículo:",
        "deu_Latn": "Fasse diesen Artikel zusammen:",
        "jpn_Jpan": "この記事を要約してください：",
        "arb_Arab": "لخّص هذا المقال:",
        "zho_Hans": "请总结这篇文章：",
        "afr_Latn": "Som hierdie artikel op:",
        "nld_Latn": "Vat dit artikel samen:",
        "fra_Latn": "Résumez cet article :",
        "por_Latn": "Resuma este artigo:",
        "rus_Cyrl": "Сделайте краткое изложение этой статьи:",
        "kor_Hang": "이 기사를 요약하세요:",
        "hin_Deva": "इस लेख का सारांश दें:",
        "tur_Latn": "Bu makaleyi özetleyin:",
        "pol_Latn": "Streść ten artykuł:",
        "swe_Latn": "Sammanfatta denna artikel:",
        "dan_Latn": "Opsummer denne artikel:",
        "nob_Latn": "Oppsummer denne artikkelen:",
    }

    prompts2 = {
        "eng_Latn": "Summary:",
        "bod_Tibt": "མདོར་བསྡུས།",
        "mlt_Latn": "Sommarju:",
        "ita_Latn": "Riassunto:",
        "spa_Latn": "Resumen:",
        "deu_Latn": "Zusammenfassung:",
        "jpn_Jpan": "要約：",
        "arb_Arab": "الملخص:",
        "zho_Hans": "摘要：",
        "afr_Latn": "Opsomming:",
        "nld_Latn": "Samenvatting:",
        "fra_Latn": "Résumé :",
        "por_Latn": "Resumo:",
        "rus_Cyrl": "Резюме:",
        "kor_Hang": "요약:",
        "hin_Deva": "सारांश:",
        "tur_Latn": "Özet:",
        "pol_Latn": "Podsumowanie:",
        "swe_Latn": "Sammanfattning:",
        "dan_Latn": "Opsummering:",
        "nob_Latn": "Sammendrag:",
    }

    return prompts[target_lang], prompts2[target_lang]



crosssum_lang_map = {
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






def post_process_text(text):
    
    text = text.split("<|end_of_text|>")[0] if len(text.split("<|end_of_text|>")) > 0 else text
    #text = text.split("?")[0] if len(text.split("?")) > 0 else text
    text = text.split("\n")[0] if len(text.split("\n")) > 0 else text
    #text = text.split(".")[0] if len(text.split(".")) > 0 else text
    # Strip excess whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    #text = re.sub(r'[?]+', '?', text).strip()
    text = re.sub(r'(\?){2,}', '?', text).strip()

    return text



def format_prompt(item: Dict, source_lang: str, target_lang: str) -> str:
    """Format a FLORES-200 item into a translation prompt"""
    article = item["text"]  # source article
    reference_summary = item["summary"]  # target summary



    prompt_tmp1, prompt_tmp2 = get_language_prompts(lang_to_code(target_lang))
    prompt = f'{prompt_tmp1} "{article}".\n{prompt_tmp2}'    
    
    return prompt




def get_prompts(langs,split: str = "test", data_path: str = None, n: int = 200) -> List[Dict]:

    from datasets import load_dataset

    """Load BELEBELE dataset for reading comprehension from HuggingFace"""
    source_lang,target_lang = langs



    import json 
    with open(f"data/CrossSum-parallel.json", "r" , encoding="utf8") as f:
        dataset = {"test":json.load(f)[target_lang]}


    # source_lang = crosssum_lang_map[source_lang]
    # target_lang = crosssum_lang_map[target_lang]
    # print(source_lang,target_lang, f"{lang_to_code(source_lang)}-{lang_to_code(target_lang)}")

    # input("here:")
    # dataset = load_dataset(f"csebuetnlp/CrossSum", f"{source_lang}-{target_lang}", trust_remote_code=True)
    # print(dataset[split])
    

    questions_list = []
    for item in dataset[split]:
        source_text = item['text']
        ref = item['summary']

        prompt = format_prompt(dict(item), source_lang, target_lang) 

        questions_list.append((prompt,ref,source_text))
    # questions_dict[source_lang] = questions_list

        # except Exception as e:
        #     print(f"Error loading BELEBELE data for language {source_lang}: {e}")

    return questions_list #[:20]