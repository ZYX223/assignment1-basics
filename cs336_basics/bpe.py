import regex as re
from collections import defaultdict
from tqdm.contrib.concurrent import process_map
import time
from itertools import pairwise
import yaml, json
from typing import Iterator, Iterable

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def read_text(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def split_by_special(text, special_tokens, drop_special=True):
    if not special_tokens:
        return [text]

    # Sort by descending length to prioritize longer tokens (e.g., "<|endoftext|><|endoftext|>" before "<|endoftext|>")
    special_tokens = sorted(special_tokens, key=len, reverse=True)

    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    if not drop_special: pattern = f"({pattern})"

    pattern = re.compile(pattern)
    chunks = pattern.split(text)
    return [c for c in chunks if c]  # remove empty strings

def word2bytes(word):
    "Convert word string to tuple of bytes"
    a = list(word.encode('utf-8'))
    return tuple(bytes([i]) for i in a)

def count_word(text):
    "Split text into word bytes using GPT2 pattern and count word bytes frequency."
    word_cnt = defaultdict(int)
    for m in PAT.finditer(text):
        word = m.group(0)
        word_bytes = word2bytes(word)
        if len(word_bytes)>=2:
            word_cnt[word_bytes]+=1
    return word_cnt

def merge_dicts(dicts):
    merged = defaultdict(int)
    for d in dicts:
        for k, v in d.items():
            merged[k] += v
    return merged

def count_pair(word_cnt):
    pair_cnt = defaultdict(int)
    for word_bytes,cnt in word_cnt.items():
        for pair in zip(word_bytes[:-1],word_bytes[1:]):
            pair_cnt[pair]+=cnt
    return pair_cnt

def get_max_pair(pair_cnt):
    max_pair, _ = max(pair_cnt.items(), key=lambda x: (x[1], x[0]))  # lexicographic tie-breaker
    return max_pair


def get_basic_vocab(special_tokens):
    # Initial vocab: all byte values + special tokens
    vocab={token:bytes([token]) for token in range(256)}
    for i,token in enumerate(special_tokens):
        token_id = 256+i
        vocab[token_id] = token.encode("utf-8")
    return vocab


def apply_merge(word_bytes,merge):
    merged = merge[0]+merge[1]
    i = 0
    new_word_bytes = []
    while i < len(word_bytes):
        # Check for match
        if i < len(word_bytes) - 1 and word_bytes[i] == merge[0] and word_bytes[i+1] == merge[1]:
            new_word_bytes.append(merged)
            i += 2
        else:
            new_word_bytes.append(word_bytes[i])
            i += 1
    return tuple(new_word_bytes)

def update_cnt(word_cnt,pair_cnt, merge_pair):

    new_word_cnt = defaultdict(int)
    new_pair_cnt = defaultdict(int, pair_cnt) # copy with defaultdict

    for word_bytes,cnt in word_cnt.items():

        #----------for word cnt ---------------

        old_pairs = list(zip(word_bytes[:-1], word_bytes[1:]))

        # Keep the original count if the merge not appear in the key
        if merge_pair not in old_pairs:
            new_word_cnt[word_bytes]+=cnt
            continue

        # Use updated key if merge appear
        new_word = apply_merge(word_bytes,merge_pair)
        new_word_cnt[new_word]+=cnt

        #--------for pair cnt ----------------

        # Decrease all old pair counts
        for pair in old_pairs:
            new_pair_cnt[pair]-=cnt
            if new_pair_cnt[pair] ==0:
                del new_pair_cnt[pair]

        # Count new pairs in the new word
        new_pairs = list(zip(new_word[:-1], new_word[1:]))
        for p in new_pairs:
            new_pair_cnt[p] += cnt

    return new_word_cnt,new_pair_cnt

def update_cnt_fast(word_cnt, pair_cnt, merge_pair):
    a, b = merge_pair
    new_word_cnt = defaultdict(int)
    # Slightly faster to modify in place, but more error-prone
    # new_pair_cnt = defaultdict(int, pair_cnt)  # copy

    for wbytes, cnt in word_cnt.items():
        # cheap presence check (no list/zip)
        has = False
        i, n = 0, len(wbytes) - 1
        while i < n:
            if wbytes[i] == a and wbytes[i+1] == b:
                has = True
                break
            i += 1
        
        if not has:
            new_word_cnt[wbytes] += cnt
            continue

        # decrement old pairs (iterator, no list)
        for p in pairwise(wbytes):
            v = pair_cnt[p] - cnt
            if v: pair_cnt[p] = v
            else: pair_cnt.pop(p, None)

        # merge & add new pairs
        new_w = apply_merge(wbytes, merge_pair)
        new_word_cnt[new_w] += cnt
        for p in pairwise(new_w):
            pair_cnt[p] += cnt

    return new_word_cnt, pair_cnt


def train_bpe(input_path,vocab_size,special_tokens):

    text = read_text(input_path)
    chunks = split_by_special(text,special_tokens)

    # Only parallelize if chunk count is big enough
    print("num chunks:",len(chunks))
    if len(chunks) < 4: word_dicts = list(map(count_word, chunks))
    else: word_dicts = process_map(count_word, chunks, chunksize=100,max_workers=16)
  
    word_cnt = merge_dicts(word_dicts)
    pair_cnt = count_pair(word_cnt)

    vocab = get_basic_vocab(special_tokens)
    base_vocab_size = len(vocab)
    n_merges=vocab_size-base_vocab_size 
    
    merges = []
    part1=0.0 ; part2 = 0.0
    for i in range(n_merges):
        t1 = time.time()
        max_pair = get_max_pair(pair_cnt)
        t2 = time.time()
        part1 += t2-t1
        vocab[base_vocab_size+i] = max_pair[0]+max_pair[1]
        merges.append(max_pair)
        # word_cnt, pair_cnt = update_cnt(word_cnt,pair_cnt,max_pair)
        word_cnt, pair_cnt = update_cnt_fast(word_cnt, pair_cnt, max_pair)
        t3 = time.time()
        part2 += t3-t2
    
    print("bpe part time",part1,part2)
    return vocab, merges

# save/load vocab and merges to/from YAML file
def save_tokenizer_yaml(vocab, merges, fname):
    "Save vocab and merges to a YAML file with UTF-8 decoding for readability."
    # Convert bytes → string for readability
    vocab_serializable = {
        k: v.decode("utf-8", errors="replace") if isinstance(v, bytes) else v
        for k, v in vocab.items()
    }
    merges_serializable = [
        (a.decode("utf-8", errors="replace"), b.decode("utf-8", errors="replace"))
        for a, b in merges
    ]
    
    with open(fname, "w", encoding="utf-8") as f:
        yaml.dump(
            {"vocab": vocab_serializable, "merges": merges_serializable},
            f,
            allow_unicode=True,
            sort_keys=False
        )

def load_tokenizer_yaml(fname):
    "Load vocab and merges from a YAML file, converting strings back to bytes."
    with open(fname, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    vocab_loaded = {
        int(k): v.encode("utf-8") if isinstance(v, str) else v
        for k, v in data["vocab"].items()
    }
    merges_loaded = [
        (a.encode("utf-8"), b.encode("utf-8")) for a, b in data["merges"]
    ]
    return vocab_loaded, merges_loaded

def split_to_words(text):
    "Split text into words."
    return re.findall(PAT,text)

def apply_merges(word_bytes, merges, vocab_to_id):
    "Apply merges based on minimum vocab token id."
    while True:
        pairs = list(zip(word_bytes[:-1], word_bytes[1:]))

        # Collect valid merge candidates with their vocab ID
        candidates = {}
        for pair in pairs:
            if pair in merges:
                merged = pair[0] + pair[1]
                token_id = vocab_to_id.get(merged) 
                if token_id is not None:
                    candidates[pair] = token_id

        if not candidates:
            break  # no more mergeable pairs

        # Choose the pair with the **smallest token ID**
        best_pair = min(candidates.items(), key=lambda x: x[1])[0]

        word_bytes = apply_merge(word_bytes, best_pair)

    return word_bytes

def encode_merged(text,merges,vocab_to_id):
    word_list = split_to_words(text)
    tokens=[]
    for word in word_list:
        word_bytes=word2bytes(word)
        merged_word_bytes = apply_merges(word_bytes,merges,vocab_to_id)
        tokens+=[vocab_to_id[i] for i in merged_word_bytes]
    return tokens

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
        
        self.vocab_to_id={v:k for k,v in vocab.items()}
        # Ensure special tokens are in the vocabulary
        for token_bytes in self.special_tokens:
            if token_bytes not in self.vocab_to_id:
                # Add to vocab if not already present
                new_id = len(self.vocab)
                self.vocab[new_id] = token_bytes
                self.vocab_to_id[token_bytes] = new_id

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        # Load vocab (assumed to be a JSON file: {token_id: byte_string})
        with open(vocab_filepath, 'r', encoding='utf-8') as vf:
            vocab_data = json.load(vf)
            # Optional: convert keys to int if stored as strings
            vocab = {int(k): bytes(v, 'latin1') if isinstance(v, str) else bytes(v)
                     for k, v in vocab_data.items()}

        # Load merges (assumed to be a list of pairs like: "a b")
        with open(merges_filepath, 'r', encoding='utf-8') as mf:
            lines = mf.readlines()
            # Optional: skip headers like "#version: 0.2"
            merge_pairs = [tuple(line.strip().split()) for line in lines if not line.startswith('#') and line.strip()]
            # Convert to byte-pairs
            merges = [(a.encode('utf-8'), b.encode('utf-8')) for a, b in merge_pairs]

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)


    def encode(self, text: str) -> list[int]:
        chunks = split_by_special(text, self.special_tokens, drop_special=False)
        tokens = []
        for chunk in chunks:
            if self.special_tokens and chunk in self.special_tokens:
                tokens.append(self.vocab_to_id[chunk.encode('utf-8')])
            else:
                tokens.extend(encode_merged(chunk, self.merges, self.vocab_to_id))
        return tokens
    

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs.
        This is required for memory-efficient tokenization of large files that we cannot directly load into memory.
        """
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        "Decode a sequence of token IDs into text."
        return b''.join([self.vocab[t] for t in ids]).decode('utf-8',errors='replace')