import argparse

from detection import euphemism_detection, evaluate_detection
from identification import euphemism_identification
from read_file import read_all_data
from predict import euphemism_predict
import torch
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', type=str, default="weapon_corpus")  # dataset file name
parser.add_argument("--dataset", type=str, default="reddit_corpus")
parser.add_argument("--target", type=str, default="drug")  # [drug, weapon, sex]
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--c1", type=int, default=2)
parser.add_argument("--c2", type=int, default=4)
parser.add_argument("--coarse", type=int, default=1)
args = parser.parse_args()
# torch.cuda.set_device(1)

""" Read Data """
all_text, euphemism_answer, input_keywords, target_name, audios, imgs = read_all_data(
    args.dataset, args.target
)
# for key in audios.keys():
#     print(key, audios[key].shape, imgs[key].shape)

# Euphemism Detection
# all_top_words = set()
# for _ in range(10):
#     top_words = euphemism_detection(
#         input_keywords, all_text, ms_limit=2000, filter_uninformative=1
#     )
#     all_top_words.update(top_words)
top_words = euphemism_detection(
    input_keywords, all_text, ms_limit=2000, filter_uninformative=1
)
# filepath = f"./data/updated_euphemism_answer_{args.target}.txt"
# euph_words = []
# with open(filepath, "r", encoding="utf-8") as f:
#     for line in f:
#         line = line.strip().split("; ")
#         euph_words.extend(line)
# # print(euph_words)
# euph_words_set = set(euph_words)
# intersection = all_top_words.intersection(euph_words_set)

# # 将这些行写入文件
# with open(f"./data/output_{args.target}.txt", "w") as f:
#     count = 0
#     for word in intersection:
#         f.write(word + "; ")
#         count += 1
#         if count % 8 == 0:
#             f.write("\n")
evaluate_detection(top_words, euphemism_answer)

# Euphemism Identification
euphemism_identification(
    top_words,
    all_text,
    euphemism_answer,
    input_keywords,
    target_name,
    args,
    imgs,
    audios,
    args.target,
    args.batch_size,
)

# for num in range(100):
#     euphemism_predict(
#         top_words,
#         all_text,
#         euphemism_answer,
#         input_keywords,
#         target_name,
#         args,
#         imgs,
#         audios,
#         args.target,
#         args.batch_size,
#     )
#     print("End of " + str(num + 1) + " cycle")
