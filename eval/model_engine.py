class PrefixLMEngine:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer



    def generate(self, prompts, suffix, prefixlm_mode, **generate_kwargs):
        tok_data = self.tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left").to(self.model.device)
        input_ids = tok_data.input_ids
        attention_mask = tok_data.attention_mask

        prompt_mask = attention_mask.clone()
        prompt_mask[prompt_mask == 0] = -100
        suffix_len = input_ids.shape[1] - self.tokenizer(prompts[0][:-len(suffix)], return_tensors="pt").input_ids.shape[1]
        prompt_mask[:, -suffix_len:] = 0
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt_mask": prompt_mask,
        }
        outputs = self.model.generate(**inputs, **generate_kwargs)

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)