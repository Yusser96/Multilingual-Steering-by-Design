

from datasets import load_dataset
import os
import json



def get_flores_language_mapping():
    """
    Mapping from 2-letter ISO 639-1 codes to FLORES-200 codes used in the original script.
    """
    return {
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
    

def load_flores_data(split: str = "dev", data_path: str = None):
    """Load FLORES-200 dataset for translation from HuggingFace"""
    # Load the dataset from HuggingFace

    lang_codes_mapping = get_flores_language_mapping()
    
    data = {}

    source_lang = "eng_Latn"
    source_lang_k = "en"

    # print('" "'.join(list(lang_codes_mapping.values())))
    # return ''
    for tgt_lang_k, tgt_lang in zip(lang_codes_mapping.keys(),lang_codes_mapping.values()):
        try:
            dataset = load_dataset("facebook/flores", name=f"{source_lang}-{tgt_lang}", trust_remote_code=True)

            if source_lang not in data:
                data[source_lang_k] = [{"prompt": item['sentence_' + source_lang]} for item in dataset[split]]

            if tgt_lang not in data:
                data[tgt_lang_k] = [{"prompt": item['sentence_' + tgt_lang]} for item in dataset[split]]
        except:
            print(f"Error with: {source_lang}-{tgt_lang}")


    os.makedirs("data",exist_ok=True)
    output_path = "data/flores200_dataset_low_res.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)





load_flores_data()


