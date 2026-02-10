from datasets import load_dataset

from math import exp
from typing import Dict, List, Tuple
from typing import List, Optional
from jaxtyping import Float, Int
import torch
from tqdm import tqdm
import json


_model_output = "text"
_max_new_tokens = 64



# device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")



import re


def init(device):
    import fasttext


    try:
        langid_model = fasttext.load_model('lid218e.bin')
    except:
        import urllib.request
        urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/nllb/lid/lid218e.bin', 'lid218e.bin')
        langid_model = fasttext.load_model('lid218e.bin')


    from comet import download_model, load_from_checkpoint
    # Download and load the COMET model

    model_path = download_model("Unbabel/wmt22-comet-da") #"wmt21-cometinho-da")
    comet_model = load_from_checkpoint(model_path).to(device)


    return langid_model, comet_model

def compute_comet_score(comet_model, sources, references, translations,batch_size=32,device=None):
    """
    Evaluates translations against reference translations using the COMET model.

    Args:
    sources (list of str): The source sentences.
    references (list of str): The reference translations.
    translations (list of str): The machine translations to evaluate.

    Returns:
    list of float: The COMET scores for each translation.
    """
    # Prepare the data as a list of dictionaries
    data = [
        {"src": src, "mt": mt, "ref": ref}
        for src, mt, ref in zip(sources, translations, references)
    ]

    # Predict the scores
    # results = comet_model.predict(data, batch_size=batch_size,accelerator="gpu", gpus=1 if device == "cuda" else 0)
    results = comet_model.predict(
    data,
    batch_size=batch_size,
    accelerator="gpu",
    # devices=1,
)
    return results['system_score']*100


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


def compute_bleu_score(model, reference: str, candidate: str) -> float:
    """Compute BLEU score between reference and candidate translations"""
    #from sacrebleu.metrics.bleu import _get_tokenizer, _TOKENIZERS

    import sacrebleu
    from functools import lru_cache

    class MyTok(sacrebleu.tokenizers.tokenizer_base.BaseTokenizer):

        def signature(self):
            return 'mytok'

        @lru_cache(maxsize=2**16)
        def __call__(self, line):
            tokenizer = model.get_tokenizer()
            tokens = tokenizer.tokenize(line, add_special_tokens=False)
            return " ".join(tokens)
    

    def my_get_tokenizer(name): 
        return MyTok
    sacrebleu.metrics.bleu._get_tokenizer = my_get_tokenizer

    #from sacrebleu import sentence_bleu
    score = sacrebleu.sentence_bleu(candidate, [reference],tokenize="mytok" )
    
    return score.score


def get_language_prompts(target_lang):
    """Target-language prompts equivalent to English 'Translate this sentence:'"""
    prompts = {
        "eng_Latn": "Translate this sentence:",
        "bod_Tibt": "ཚིག་འདི་སྐད་ཡིག་སྒྱུར་གནང་:",
        "mlt_Latn": "Ittraduċi din is-sentenza:",
        "ita_Latn": "Traduci questa frase:",
        "spa_Latn": "Traduce esta frase:",
        "deu_Latn": "Übersetze diesen Satz:",
        "jpn_Jpan": "この文を翻訳してください：",
        "arb_Arab": "ترجم هذه الجملة:",
        "zho_Hans": "请翻译这句话：",
        "afr_Latn": "Vertaal hierdie sin:",
        "nld_Latn": "Vertaal deze zin:",
        "fra_Latn": "Traduisez cette phrase :",
        "por_Latn": "Traduza esta frase:",
        "rus_Cyrl": "Переведите это предложение:",
        "kor_Hang": "이 문장을 번역하세요:",
        "hin_Deva": "इस वाक्य का अनुवाद करें:",
        "tur_Latn": "Bu cümleyi çevirin:",
        "pol_Latn": "Przetłumacz to zdanie:",
        "swe_Latn": "Översätt den här meningen:",
        "dan_Latn": "Oversæt denne sætning:",
        "nob_Latn": "Oversett denne setningen:",
    }

    prompts2 = {
        "eng_Latn": "Translation:",
        "bod_Tibt": "ཡིག་སྒྱུར་:",
        "mlt_Latn": "Traduzzjoni:",
        "ita_Latn": "Traduzione:",
        "spa_Latn": "Traducción:",
        "deu_Latn": "Übersetzung:",
        "jpn_Jpan": "翻訳：",
        "arb_Arab": "الترجمة:",
        "zho_Hans": "翻译：",
        "afr_Latn": "Vertaling:",
        "nld_Latn": "Vertaling:",
        "fra_Latn": "Traduction :",
        "por_Latn": "Tradução:",
        "rus_Cyrl": "Перевод:",
        "kor_Hang": "번역:",
        "hin_Deva": "अनुवाद:",
        "tur_Latn": "Çeviri:",
        "pol_Latn": "Tłumaczenie:",
        "swe_Latn": "Översättning:",
        "dan_Latn": "Oversættelse:",
        "nob_Latn": "Oversettelse:",
    }

    return prompts[target_lang], prompts2[target_lang]

def get_language_names():
    """Human-readable language names for FLORES-200 codes"""
    return {
        "bod_Tibt": "Tibetan",
        "mlt_Latn": "Maltese", 
        "ita_Latn": "Italian",
        "spa_Latn": "Spanish",
        "deu_Latn": "German",
        "jpn_Jpan": "Japanese",
        "arb_Arab": "Arabic",
        "zho_Hans": "Chinese",
        "afr_Latn": "Afrikaans",
        "nld_Latn": "Dutch",
        "fra_Latn": "French",
        "por_Latn": "Portuguese",
        "rus_Cyrl": "Russian",
        "kor_Hang": "Korean",
        "hin_Deva": "Hindi",
        "tur_Latn": "Turkish",
        "pol_Latn": "Polish",
        "swe_Latn": "Swedish",
        "dan_Latn": "Danish",
        "nob_Latn": "Norwegian",
        "eng_Latn": "English"
    }

def get_language_mapping():
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
# "ar" "de" "es" "fr" "ru" "hi" "tr" "en" "zh"
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
    
def load_flores_data( target_lang: str,source_lang: str  = "en", split: str = "test", data_path: str = None, n: int = 200) -> List[Dict]:
    """Load FLORES-200 dataset for translation from HuggingFace"""
    # Load the dataset from HuggingFace
    dataset = load_dataset("facebook/flores", name=f"{source_lang}-{target_lang}", trust_remote_code=True)
    
    data = []
    for item in dataset[split]:
        data.append(dict(item))
    
    return data[:n]

def format_translation_prompt(item: Dict, source_lang: str, target_lang: str, lang_name_mapping: Dict) -> str:
    """Format a FLORES-200 item into a translation prompt"""
    source_text = item['sentence_' + source_lang]
    
    # Get human-readable language names
    source_name = lang_name_mapping.get(source_lang, source_lang)
    target_name = lang_name_mapping.get(target_lang, target_lang)
    
    #prompt = f"Translate this {source_name} sentence into {target_name}: {source_text}. Translation:"
    prompt_tmp1, prompt_tmp2 = get_language_prompts(target_lang)
    prompt = f'{prompt_tmp1} "{source_text}".\n{prompt_tmp2}'
    
    
    return prompt



def get_prompts(langs,split: str = "devtest", data_path: str = None, n: int = 200) -> List[Dict]:
    """Load BELEBELE dataset for reading comprehension from HuggingFace"""
    source_lang,target_lang = langs
    source_lang = lang_to_code(source_lang)
    target_lang = lang_to_code(target_lang)
    # print(source_lang,target_lang, f"{lang_to_code(source_lang)}-{lang_to_code(target_lang)}")

    # input("here:")
    dataset = load_dataset("facebook/flores", name=f"{source_lang}-{target_lang}", trust_remote_code=True)
    
    lang_name_mapping = get_language_names()

    questions_list = []
    for item in dataset[split]:
        source_text = item['sentence_' + source_lang]
        ref = item['sentence_' + target_lang]

        prompt = format_translation_prompt(dict(item), source_lang, target_lang, lang_name_mapping) 

        questions_list.append((prompt,ref,source_text))
    # questions_dict[source_lang] = questions_list

        # except Exception as e:
        #     print(f"Error loading BELEBELE data for language {source_lang}: {e}")

    return questions_list #[:20]










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

