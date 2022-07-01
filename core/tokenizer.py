from pathlib import Path
import torch
from tokenizers import Tokenizer
from tokenizers.processors import ByteLevel

# Source: https://github.com/lucidrains/DALLE-pytorch
class HugTokenizer:
    def __init__(self, bpe_path, text_len, truncate_text=True):
        bpe_path = Path(bpe_path)
        assert bpe_path.exists(), f'BPE json path {str(bpe_path)} does not exist'
        tokenizer = Tokenizer.from_file(str(bpe_path))
        tokenizer.post_processor = ByteLevel(trim_offsets = True)
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()
        self.text_len = text_len
        self.truncate_text = truncate_text

    def decode(self, tokens, pad_tokens = {}):
        if torch.is_tensor(tokens):
            tokens = tokens.tolist()
        #ignore_ids = pad_tokens.union({0})
        #tokens = [token for token in tokens if token not in ignore_ids]
        return self.tokenizer.decode(tokens, skip_special_tokens = True)

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        all_tokens = [self.encode(text) for text in texts]

        result = torch.zeros(len(all_tokens), self.text_len, dtype=torch.long)
        for i, tokens in enumerate(all_tokens):
            if len(tokens) > self.text_len:
                if self.truncate_text:
                    tokens = tokens[:self.text_len]
                else:
                    raise RuntimeError(f"Input {texts[i]}'s length {len(tokens)} is too long for context length {text_len}")
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result
