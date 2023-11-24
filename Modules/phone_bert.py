import torch
from transformers import AutoModel, AutoTokenizer
from text2phonemesequence import Text2PhonemeSequence


pretrained_path = "/Users/zhangsan/Downloads/vinai-xphonebert-base"
# Load XPhoneBERT model and its tokenizer
xphonebert = AutoModel.from_pretrained(pretrained_path)
tokenizer = AutoTokenizer.from_pretrained(pretrained_path)

# Load Text2PhonemeSequence
# text2phone_model = Text2PhonemeSequence(language='eng-us', is_cuda=True)
pretrained_g2p_model_path = "/Users/zhangsan/Downloads/charsiu-g2p_multilingual_byT5_small_100"
tokenizer_path = "/Users/zhangsan/Downloads/google-byt5-small"
text2phone_model = Text2PhonemeSequence(pretrained_g2p_model=pretrained_g2p_model_path, tokenizer=tokenizer_path, language='zho-t', is_cuda=False)

# Input sequence that is already WORD-SEGMENTED (and text-normalized if applicable)
# sentence = "That is , it is a testing text ."
sentence = "今天是一个风和日丽的日子，去游泳吧。"

input_phonemes = text2phone_model.infer_sentence(sentence)

input_ids = tokenizer(input_phonemes, return_tensors="pt")

with torch.no_grad():
    features = xphonebert(**input_ids)
print("Done")
