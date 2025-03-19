# Euphemism_Identification# 多模态委婉语识别项目

这是一个基于多模态数据的委婉语分析项目，包含委婉语检测和识别功能。

## 项目结构

```
code/
  ├── Main.py           # 主程序入口
  ├── read_file.py      # 数据读取模块
  ├── detection.py      # 委婉语检测模块
  ├── identification.py # 委婉语识别模块
  └── classification_model.py # 分类模型定义
```

## 环境依赖

项目主要依赖以下Python包：

```
torch
transformers
librosa
Pillow
tqdm
torchvggish
numpy
```

同时需要以下预训练模型：
- ViT 
- BERT
- Wav2Vec2

## 数据准备

1. 在项目根目录下创建以下数据目录结构：
```
data/
  ├── text/            # 存放文本数据
  ├── euphemism_answer_{target_category_name}.txt  # 委婉语答案文件
  └── target_keywords_{target_category_name}.txt   # 目标关键词文件

new-audio/            # 音频数据目录
  └── {target}/       # 按目标类别组织的音频文件

new-img/              # 图像数据目录
  └── {target}/       # 按目标类别组织的图像文件
```

2. 数据格式要求：
- 文本数据：每行一个句子
- 音频文件：WAV格式，16kHz采样率
- 图像文件：PNG格式
- 委婉语答案文件格式：`答案:委婉语1;委婉语2;...`
- 目标关键词文件格式：每行用制表符分隔的同义词组

## 运行方法

1. 安装依赖：
```bash
pip install torch transformers librosa Pillow tqdm torchvggish numpy
```

2. 运行主程序：
```bash
python Main.py --dataset reddit_corpus --target drug --batch_size 128 --lr 5e-5
```

主要参数说明：
- `--dataset`: 数据集名称，默认为"reddit_corpus"
- `--target`: 目标类别，可选[drug, weapon, sex]，默认为"drug"
- `--batch_size`: 批处理大小，默认为128
- `--lr`: 学习率，默认为5e-5
- `--c1`: 参数1，默认为2
- `--c2`: 参数2，默认为4
- `--coarse`: 粗粒度参数，默认为1

## 功能模块

1. 数据读取 (read_file.py)
- 支持读取文本、音频和图像数据
- 使用预训练模型提取特征

2. 委婉语检测 (detection.py)
- 基于输入关键词检测潜在的委婉语
- 支持过滤无信息词

3. 委婉语识别 (identification.py)
- 多模态特征融合
- 基于LSTM和CNN的分类模型

4. 分类模型 (classification_model.py)
- LSTM with Attention
- CNN with multiple kernel sizes

## 注意事项

1. 确保有足够的GPU内存运行预训练模型
2. 首次运行时会下载预训练模型，需要稳定的网络连接
3. 音频和图像特征会被缓存到data目录下，可加快后续运行速度
