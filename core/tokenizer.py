from pathlib import Path
import torch
from tokenizers import Tokenizer
from tokenizers.processors import ByteLevel

class HugTokenizer:
    def __init__(self, bpe_path = None):
        bpe_path = Path(bpe_path)
        assert bpe_path.exists(), f'BPE json path {str(bpe_path)} does not exist'
        tokenizer = Tokenizer.from_file(str(bpe_path))
        tokenizer.post_processor = ByteLevel(trim_offsets = True)
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()

    def decode(self, tokens, pad_tokens = {}):
        if torch.is_tensor(tokens):
            tokens = tokens.tolist()
        #ignore_ids = pad_tokens.union({0})
        #tokens = [token for token in tokens if token not in ignore_ids]
        return self.tokenizer.decode(tokens, skip_special_tokens = True)

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def tokenize(self, texts, context_length = 256, truncate_text = False):
        if isinstance(texts, str):
            texts = [texts]

        all_tokens = [self.encode(text) for text in texts]

        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                if truncate_text:
                    tokens = tokens[:context_length]
                else:
                    raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result
