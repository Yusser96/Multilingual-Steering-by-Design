import abc 
import torch



class BaseTask():
    self._max_new_tokens = 1
    self._model_output = "text" # "logits"
    self.CHOICES = None #["A", "B", "C", "D"]


    def __init__(self):
        self.model = model
        self.tokenizer = self.model.tokenizer

    @required
    def get_prompts():
        pass
        
    @required
    def eval():
        pass


    def build_choice_token_ids(self):
        """
        Build a small map from each choice (A/B/C/D) to a set of token IDs
        that can represent that choice as the *next token*.
        """
        

        # You can expand this with more patterns if needed
        VARIANTS = { c:[f"{c}", f" {c}", f"({c})", f"{c}.", f" {c}.", f"{c})", f" {c})"]
        for c in self.CHOICES
        }

        choice_to_ids = {c: set() for c in self.CHOICES}

        for choice, var_list in VARIANTS.items():
            for v in var_list:
                ids = self.tokenizer.encode(v)
                if len(ids) == 1:
                    choice_to_ids[choice].add(ids[0])
                # If the tokenizer splits it into multiple tokens, you can optionally
                # keep the last one; often the leading space or punctuation is its own token.
                elif len(ids) > 1:
                    choice_to_ids[choice].add(ids[-1])

        # Convert sets to sorted lists for stable behavior
        choice_to_ids = {c: sorted(list(ids)) for c, ids in choice_to_ids.items()}
        return choice_to_ids

    def pick_choice_from_logprobs(self, logits):
        """
        logits: [vocab] or [1, vocab] next-token logits
        choice_token_ids: dict like {"A": [id1, id2, ...], "B": [...], ...}

        Returns:
            best_choice: "A"/"B"/"C"/"D"
            probs: dict {"A": p_A, "B": p_B, ...} (normalized over just these choices)
        """

        choice_token_ids = build_choice_token_ids()

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