import time
from collections import defaultdict
import os
from tqdm import tqdm
import librosa
from PIL import Image
from transformers import (
    ViTImageProcessor,
    ViTModel,
    CLIPProcessor,
    CLIPModel,
    Wav2Vec2Processor,
    Wav2Vec2Model,
)
import torch
import torch.nn.functional as F
from torchvggish import vggish, vggish_input
import numpy as np
import gc
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# vit_processor = ViTImageProcessor.from_pretrained("vit-base-patch16-224")
# vit_model = ViTModel.from_pretrained("vit-base-patch16-224")
clip_processor = CLIPProcessor.from_pretrained("clip-vit-large-patch14")
clip_model = CLIPModel.from_pretrained("clip-vit-large-patch14").to(device)
wav2vec_processor = Wav2Vec2Processor.from_pretrained("wav2vec2-large-960h")
wav2vec_model = Wav2Vec2Model.from_pretrained("wav2vec2-large-960h").to(device)
""" Read Data """


def read_raw_text(fname, input_keywords):
    start = time.time()
    all_text = []
    num_lines = sum(1 for line in open(fname, "r"))
    with open(fname, "r") as fin:
        for line in tqdm(fin, total=num_lines):
            temp = line.split()
            if any(ele in temp for ele in input_keywords) and len(line) <= 150:
                all_text.append(line.strip())
    print("[read_data.py] Finish reading data using %.2fs" % (time.time() - start))
    return all_text


def read_input_and_ground_truth(target_category_name):
    fname_euphemism_answer = "./data/euphemism_answer_" + target_category_name + ".txt"
    fname_target_keywords_name = (
        "./data/target_keywords_" + target_category_name + ".txt"
    )
    euphemism_answer = defaultdict(list)
    with open(fname_euphemism_answer, "r") as fin:
        for line in fin:
            ans = line.split(":")[0].strip().lower()
            for i in line.split(":")[1].split(";"):
                euphemism_answer[i.strip().lower()].append(ans)
    input_keywords = sorted(
        list(set([y for x in euphemism_answer.values() for y in x]))
    )
    target_name = {}
    count = 0
    with open(fname_target_keywords_name, "r") as fin:
        for line in fin:
            for i in line.strip().split("\t"):
                target_name[i.strip()] = count
            count += 1
    return euphemism_answer, input_keywords, target_name


def read_audio(tgt):
    def mean_pooling(model_output, attention_mask):
        # 提取最后一层的特征和注意力掩码
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    audio_dict = {}
    if os.path.exists(f"./data/{tgt}_audio_dict_2dim.pkl"):
        with open(f"./data/{tgt}_audio_dict_2dim.pkl", "rb") as f:
            audio_dict = pickle.load(f)
        return audio_dict
    folder_path = f"./new-audio/{tgt}/"
    audio_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".wav")
    ]
    for filename in tqdm(audio_files):
        if filename.endswith("_0.wav"):
            continue
        name, ext = os.path.splitext(filename)
        waveform, sr = librosa.load(filename, sr=16000)
        inputs = wav2vec_processor(
            waveform, sampling_rate=16000, return_tensors="pt", padding=True
        )
        inputs = inputs.input_values.to(device)
        with torch.no_grad():
            wav2vec_model.eval()
            outputs = wav2vec_model(inputs)
            attention_mask = outputs.last_hidden_state.ne(0)
            # 使用平均池化获取固定大小的表示
            pooled_features = mean_pooling(outputs, attention_mask)
        name = name.split("_")[0].split(folder_path)[1]
        audio_dict[name] = pooled_features.cpu().squeeze() # [768]
    return audio_dict


def read_img(tgt):
    img_dict = {}
    if os.path.exists(f"./data/{tgt}_img_dict_2dim.pkl"):
        with open(f"./data/{tgt}_img_dict_2dim.pkl", "rb") as f:
            img_dict = pickle.load(f)
        return img_dict
    folder_path = f"./new-img/{tgt}/"
    img_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".png")
    ]
    for filename in tqdm(img_files):
        if filename.endswith("_0.png"):
            continue
        name, ext = os.path.splitext(filename)
        image = Image.open(filename)
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        outputs = clip_model.get_image_features(**inputs)
        features = outputs
        name = name.split("_")[0].split(folder_path)[1]
        img_dict[name] = features.cpu().detach()  # [768]
    return img_dict


def read_all_data(dataset_name, target_category_name):
    """target_name is a dict (key: a target keyword, value: index). This is for later classification purpose, since
    different target keyword can refer to the same concept (e.g., 'alprazolam' and 'xanax', 'ecstasy' and 'mdma').
    """
    print("[read_data.py] Reading data...")
    euphemism_answer, input_keywords, target_name = read_input_and_ground_truth(
        target_category_name
    )
    all_text = read_raw_text("./data/text/" + dataset_name + ".txt", input_keywords)
    audios = read_audio(target_category_name)
    imgs = read_img(target_category_name)
    return all_text, euphemism_answer, input_keywords, target_name, audios, imgs
