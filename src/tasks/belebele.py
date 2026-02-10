from datasets import load_dataset

from math import exp
from typing import Dict, List, Tuple
from typing import List, Optional
from jaxtyping import Float, Int
import torch
from tqdm import tqdm
import json

_model_output = "logits"
_max_new_tokens = "1"


def get_language_names():
    """Human-readable language names for BELEBELE codes"""
    return {
        "bo": "Tibetan",
        "mt": "Maltese", 
        "it": "Italian",
        "es": "Spanish",
        "de": "German",
        "ja": "Japanese",
        "ar": "Arabic",
        "zh": "Chinese",
        "af": "Afrikaans",
        "nl": "Dutch",
        "fr": "French",
        "pt": "Portuguese",
        "ru": "Russian",
        "ko": "Korean",
        "hi": "Hindi",
        "tr": "Turkish",
        "pl": "Polish",
        "sv": "Swedish",
        "da": "Danish",
        "no": "Norwegian",
        "en": "English"
    }

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

def get_prompt(item: Dict, source_lang: str) -> str:
    """Format a BELEBELE item into a multiple choice reading comprehension prompt"""

    lang_name_mapping  = get_language_names()
    passage = item['flores_passage']
    question = item['question']
    
    options = [
        f"A: {item['mc_answer1']}",
        f"B: {item['mc_answer2']}",
        f"C: {item['mc_answer3']}",
        f"D: {item['mc_answer4']}"
    ]
    
    language_name = lang_name_mapping.get(source_lang, source_lang)
    options = '\n'.join(options)
    prompt = f"""{passage}\n\nQ: {question}\n{options}\nAnswer:"""

    return prompt



def get_prompts(source_lang,split: str = "test", data_path: str = None, n: int = 200) -> List[Dict]:
    """Load BELEBELE dataset for reading comprehension from HuggingFace"""

    # Language codes that should match your file names (e.g., af.txt, bo.txt, etc.)
    
        # try:
        
    dataset = load_dataset("facebook/belebele", lang_to_code(source_lang))
    questions_list = []
    for item in dataset[split]:

        prompt = get_prompt(item,source_lang)
        

        num_to_choice = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
        ref = item['correct_answer_num']
        if isinstance(ref, str) and ref.isdigit():
            ref = num_to_choice.get(int(ref))
        elif isinstance(ref, int):
            ref = num_to_choice.get(ref)

        questions_list.append((prompt,ref))
    # questions_dict[source_lang] = questions_list

        # except Exception as e:
        #     print(f"Error loading BELEBELE data for language {source_lang}: {e}")

    return questions_list #[:20]


# lang_codes = [   
#         "bo", "mt", "it", "es", "de", "ja", "ar", "zh", "af", "nl", 
#         "fr", "pt", "ru", "ko", "hi", "tr", "pl", "sv", "da", "no", "en"
#     ]
# get_test_questions(lang_codes)



# def pick_choice_from_logprobs(logits,  tokenizer=None):
#     CHOICES = ["A", "B", "C", "D"]

#     def _normalize_token(tok: str) -> str:
#         t = tok.replace("▁", "").replace("Ġ", "").strip().upper()
#         t = t.replace("(", "").replace(")", "").replace(".", "").replace(":", "")
#         return t.replace(" ", "")

#     # Convert logits → probabilities
#     logprobs = torch.log_softmax(logits, dim=-1)[0]     # shape: [vocab]
#     probs = torch.exp(logprobs)

    

    
#     scores = {c: 0.0 for c in CHOICES}

#     vocab_size = probs.shape[0]

#     # Loop over every token in vocab and map probabilities to {A,B,C,D}
#     for tok_id in range(vocab_size):

#         tok_str = tokenizer.decode([tok_id])
#         letter = _normalize_token(tok_str)

#         if letter in scores:
#             scores[letter] += float(probs[tok_id].item())


#     # Normalize scores
#     total = sum(scores.values())
#     if total > 0:
#         scores = {k: v / total for k, v in scores.items()}

#     best = max(scores.items(), key=lambda kv: kv[1])[0]
#     return best, scores




def build_choice_token_ids(tokenizer):
    """
    Build a small map from each choice (A/B/C/D) to a set of token IDs
    that can represent that choice as the *next token*.
    """
    CHOICES = ["A", "B", "C", "D"]

    # You can expand this with more patterns if needed
    VARIANTS = {
        "A": ["A", " A", "(A)", "A.", " A.", "A)", " A)", "A:", " A:"],
        "B": ["B", " B", "(B)", "B.", " B.", "B)", " B)", "A:", " A:"],
        "C": ["C", " C", "(C)", "C.", " C.", "C)", " C)", "A:", " A:"],
        "D": ["D", " D", "(D)", "D.", " D.", "D)", " D)", "A:", " A:"],
    }

    # VARIANTS = {
    #     "A": ["A", " A"],
    #     "B": ["B", " B"],
    #     "C": ["C", " C"],
    #     "D": ["D", " D"],
    # }

    choice_to_ids = {c: set() for c in CHOICES}

    for choice, var_list in VARIANTS.items():
        for v in var_list:
            ids = tokenizer.encode(v)
            if len(ids) == 1:
                choice_to_ids[choice].add(ids[0])
            # If the tokenizer splits it into multiple tokens, you can optionally
            # keep the last one; often the leading space or punctuation is its own token.
            elif len(ids) > 1:
                choice_to_ids[choice].add(ids[-1])

    # Convert sets to sorted lists for stable behavior
    choice_to_ids = {c: sorted(list(ids)) for c, ids in choice_to_ids.items()}
    return choice_to_ids


def pick_choice_from_logprobs(logits,  tokenizer=None):
    """
    logits: [vocab] or [1, vocab] next-token logits
    choice_token_ids: dict like {"A": [id1, id2, ...], "B": [...], ...}

    Returns:
        best_choice: "A"/"B"/"C"/"D"
        probs: dict {"A": p_A, "B": p_B, ...} (normalized over just these choices)
    """

    choice_token_ids = build_choice_token_ids(tokenizer)

    # Handle [1, vocab] or [vocab]
    logits = logits.squeeze(0)          # -> [vocab]
    logprobs = torch.log_softmax(logits, dim=-1)   # [vocab]

    choice_logps = {}
    for choice, ids in choice_token_ids.items():
        if not ids:
            # No token ids mapped for this choice -> assign -inf
            choice_logps[choice] = float("-inf")
            continue

        # Gather logprobs for all token IDs corresponding to this choice
        idx = torch.tensor(ids, device=logprobs.device, dtype=torch.long)
        # Log-sum-exp over variants: log(sum_i exp(logp_i))
        choice_logps[choice] = torch.logsumexp(logprobs[idx], dim=0).item()

    # Convert aggregated logps to normalized probs over A/B/C/D
    logp_tensor = torch.tensor(list(choice_logps.values()))
    probs_tensor = torch.softmax(logp_tensor, dim=0)
    probs = {c: float(p) for c, p in zip(choice_logps.keys(), probs_tensor.tolist())}

    best_choice = max(probs.items(), key=lambda kv: kv[1])[0]
    return best_choice, probs



def eval(prompts, out, output_path, model=None, base_res=None):

    tokenizer = model.tokenizer

    all_results = []

    print("Eval")
    for i, ((p,c) ,l) in tqdm(enumerate(zip(prompts,out)),
            total=len(prompts)
        ):

        best_letter, score_table = pick_choice_from_logprobs(l, tokenizer=tokenizer)
        
        result = {
            "question_idx": i,
            "input": p,
            "top_lp":best_letter,
            "correct":c,
            "score_table":score_table,
        }
        all_results.append(result)

    results = {
        "results": all_results,
    }
    if all_results:
        results["accuracy"] = sum([r["correct"]==r["top_lp"] for r in all_results]) / len(all_results)


    if output_path is not None:
        results["base_results"] = base_res["results"]
        results["base_accuray"] = base_res["accuracy"]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
    print(f"Results saved to: {output_path}")

    return results