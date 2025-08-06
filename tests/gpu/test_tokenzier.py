import time
import json
from ernie.tokenizer import Ernie4_5_Tokenizer
data_file = "./examples/data/sft-train.jsonl"
tokenizer_0_3b_path = "./baidu/ERNIE-4.5-0.3B-Paddle"
tokenizer_21b_path = "./baidu/ERNIE-4.5-21B-A3B-Paddle"
tokenizer_28b_path = "./baidu/ERNIE-4.5-VL-28B-A3B-Base-Paddle"
tokenizer_300b_path = "./baidu/ERNIE-4.5-300B-A47B-Paddle"


with open(data_file, "r", encoding="utf-8") as f:
    data_list = [json.loads(line)["src"][0] for line in f if line.strip()]

def get_tokenizer_cus_time(tokenizer_path):
    cus_time = 0
    Tokenizer = Ernie4_5_Tokenizer.from_pretrained(tokenizer_path)
    for text in data_list:
        start_time = time.time()*10000
        tokens = Tokenizer.tokenize(text)
        tokens_ids = Tokenizer.convert_tokens_to_ids(tokens)
        Tokenizer.decode(tokens_ids)
        end_time = time.time()*10000
        cus_time += (end_time - start_time)
    return cus_time/len(data_list)

def assert_tokenizer_correctness(test_io, base_io):
    """assert_tokenizer_correctness"""
    assert abs(test_io - base_io) < 0.1, f"Tokenizer I/O diff error"

def test_0_3b_tokenizer_io():
    assert_tokenizer_correctness(get_tokenizer_cus_time(tokenizer_0_3b_path) , 0.78923828125)

def test_21b_tokenizer_io():
    assert_tokenizer_correctness(get_tokenizer_cus_time(tokenizer_21b_path) , 0.78923828125)

def test_28b_tokenizer_io():
    assert_tokenizer_correctness(get_tokenizer_cus_time(tokenizer_28b_path) , 1.43587890625)

def test_300b_tokenizer_io():
    assert_tokenizer_correctness(get_tokenizer_cus_time(tokenizer_300b_path) , 0.78923828125)