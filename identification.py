from collections import defaultdict

import nltk
import numpy as np
import random
import string

import torch
import torchtext
import transformers
from scipy.special import softmax
from sklearn.metrics import confusion_matrix
from torchtext.data.utils import ngrams_iterator
from torchtext.vocab import GloVe
from tqdm import tqdm
import time
import csv
import joblib
from torch.optim import AdamW
from classification_model import *
import os
import gc

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
p = 0.6
torch.set_printoptions(profile="full")
BERT_model_pth = "data/BERT_model_reddit"
model_config = transformers.BertConfig.from_pretrained(BERT_model_pth)
bert_model = BertModel.from_pretrained(BERT_model_pth).to(device)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=3848):
        super(FocalLoss, self).__init__()
        self.alpha = torch.zeros(num_classes)
        self.alpha[0] += alpha
        self.alpha[1:] += 1 - alpha
        self.gamma = gamma

    def forward(self, logits, labels):
        logits = logits.view(-1, logits.size(-1))
        self.alpha = self.alpha.to(logits.device)
        logits_logsoft = F.log_softmax(logits, dim=1)
        logits_softmax = torch.exp(logits_logsoft)
        logits_softmax = logits_softmax.gather(1, labels.view(-1, 1))
        logits_logsoft = logits_logsoft.gather(1, labels.view(-1, 1))
        self.alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - logits_softmax), self.gamma), logits_logsoft)
        loss = torch.mul(self.alpha, loss.t())[0, :]
        return loss


def poly1_focal_loss_torch(
    logits, labels, alpha=0.25, gamma=2, num_classes=3848, epsilon=1.0
):
    focal_loss_func = FocalLoss(alpha, gamma, num_classes)
    focal_loss = focal_loss_func(logits, labels)

    p = torch.sigmoid(logits)
    labels = torch.nn.functional.one_hot(labels, num_classes)
    labels = labels.clone().detach()
    poly1 = labels * p + (1 - labels) * (1 - p)
    poly1_focal_loss = focal_loss + torch.mean(
        epsilon * torch.pow(1 - poly1, 2 + 1), dim=-1
    )
    return poly1_focal_loss


def entropy_loss(outputs):
    softmax = nn.Softmax(dim=1)
    probs = softmax(outputs)
    log_probs = torch.log(probs)
    entropy = -torch.mean(torch.sum(probs * log_probs, dim=1))
    return entropy


""" Main Function """


def euphemism_identification(
    top_words,
    all_text,
    euphemism_answer,
    input_keywords,
    target_name,
    args,
    img_dict,
    audio_dict,
    tgt,
    BATCH_SIZE,
):
    print("\n" + "*" * 40 + " [Euphemism Identification] " + "*" * 40)
    """ Construct Training Dataset"""
    since_all = time.time()  # start 1
    all_classifiers = ["LRT", "LSTM", "LSTMAtten", "CNN", "Bert_FA"]
    classifier_1 = all_classifiers[args.c1]
    NGRAMS = 1
    final_test = get_final_test(euphemism_answer, top_words, input_keywords)
    (
        train_data,
        test_data,
        final_test_data,
        train_data_pre,
        test_data_pre,
        unique_vocab_dict,
        unique_vocab_list,
    ) = get_train_test_data1(
        input_keywords, target_name, all_text, final_test, NGRAMS, train_perc=0.8
    )

    final_test_output, final_test_data_t = [], []  # For faster computation

    if args.coarse:
        print("-" * 40 + " [Coarse Binary Classifier] " + "-" * 40)
        print("Model: " + classifier_1)
        if classifier_1 in ["LRT"]:
            (
                model,
                final_out,
                final_test_output,
                final_test_data_t,
            ) = train_LRT_classifier(
                train_data_pre,
                test_data_pre,
                final_test_data,
                final_test,
                unique_vocab_dict,
                unique_vocab_list,
                target_name,
                IsPre=1,
                has_coarse=0,
                bs=32,
            )
        else:
            (
                train_iter,
                test_iter,
                Final_test_iter,
                lr,
                epoch_num,
                model,
                loss_fn,
            ) = train_initialization(
                classifier_1,
                train_data_pre,
                test_data_pre,
                final_test_data,
                target_name,
                bs=32,
                IsPre=1,
            )
            for epoch in range(epoch_num):
                train_loss, train_acc = train_model(
                    model, train_iter, loss_fn, lr, epoch, bs=32
                )
                test_loss, test_acc, _ = eval_model(model, test_iter, loss_fn, bs=32)
                print(
                    f"Epoch: {epoch + 1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:3f}, Test Acc: {test_acc:.2f}%"
                )
                _, _, final_test_output = eval_model(
                    model, Final_test_iter, loss_fn, bs=32
                )
            convert_final_test_output_to_final_out(
                final_test_output,
                target_name,
                final_test,
                torch.Tensor([i[1] for i in final_test_data]).long(),
                IsPre=1,
                has_coarse=0,
            )

    print("\n" + "-" * 40 + " [Fine-grained Multi-class Classifer] " + "-" * 40)
    classifier_2 = all_classifiers[args.c2]
    print("Model: " + classifier_2)
    if classifier_2 in ["Bert_FA"]:
        model, final_out, _, _ = train_Bert_FA(
            train_data,
            test_data,
            final_test_data,
            final_test,
            target_name,
            0,
            BATCH_SIZE,
            args.lr,
            args.coarse,
            final_test_output,
            final_test_data_t,
            img_dict,
            audio_dict,
            tgt,
        )
        get_filtered_final_out(final_out, final_test, input_keywords, target_name, tgt)
        end_all = time.time()
        time_all = (end_all - since_all) / 3600
        print(f"Total spends : {time_all: .2f} hour")
    else:
        (
            train_iter,
            test_iter,
            Final_test_iter,
            lr,
            epoch_num,
            model,
            loss_fn,
        ) = train_initialization(
            classifier_2, train_data, test_data, final_test_data, target_name, bs=32
        )
        for epoch in range(epoch_num):
            train_loss, train_acc = train_model(
                model, train_iter, loss_fn, lr, epoch, bs=32
            )
            test_loss, test_acc, _ = eval_model(model, test_iter, loss_fn, bs=32)
            print(
                f"Epoch: {epoch + 1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:3f}, Test Acc: {test_acc:.2f}%"
            )
            _, _, final_test_output = eval_model(model, Final_test_iter, loss_fn, bs=32)
        final_out = convert_final_test_output_to_final_out(
            final_test_output,
            target_name,
            final_test,
            torch.Tensor([i[1] for i in final_test_data]).long(),
            0,
            args.coarse,
            final_test_output,
        )
        get_filtered_final_out(final_out, final_test, input_keywords, target_name, tgt)
    return 0


def train_Bert_FA(
    train_data,
    test_data,
    final_test_data,
    final_test,
    target_name,
    IsPre,
    bs,
    lr,
    has_coarse=0,
    final_test_output_pre=[],
    final_test_data_t=[],
    img_dict={},
    audio_dict={},
    tgt="drug",
):
    def to_bert_input(tokens, bert_tokenizer):
        token_idx = torch.tensor(bert_tokenizer.convert_tokens_to_ids(tokens))
        sep_idx = tokens.index("[SEP]")
        segment_idx = torch.tensor(len(token_idx) * [0] + (128 - len(token_idx)) * [0])
        mask = torch.tensor(len(token_idx) * [1] + (128 - len(token_idx)) * [0])
        token_idx = torch.tensor(token_idx.tolist() + (128 - len(token_idx)) * [0])
        return (
            token_idx.unsqueeze(0).to(device),
            segment_idx.unsqueeze(0).to(device),
            mask.unsqueeze(0).to(device),
        )

    def single_bert(message):
        tokens = bert_tokenizer.tokenize(message)
        if len(tokens) == 0:
            return []
        if tokens[0] != CLS:
            tokens = [CLS] + tokens
        if tokens[-1] != SEP:
            tokens.append(SEP)
        index = tokens.index("[MASK]")
        index = torch.tensor(index).unsqueeze(0)
        token_idx, segment_idx, mask = to_bert_input(tokens, bert_tokenizer)
        return token_idx, segment_idx, mask, index

    def get_prompt(key_target):
        target_words = [key for key in target_name if target_name[key] == key_target]
        target_words = " ".join(target_words)
        # for key in target_name:
        #     if target_name[key] != key_target and target_words == "":
        #         continue
        #     elif target_name[key] == key_target:
        #         target_words = target_words + key + " "
        #         continue
        #     break
        return target_words

    def single_bert1(message):
        Grounds = get_prompt(message)
        tokens = bert_tokenizer.tokenize(Grounds)
        if len(tokens) == 0:
            return []
        if tokens[0] != CLS:
            tokens = [CLS] + tokens
        if tokens[-1] != SEP:
            tokens.append(SEP)
        token_idx = torch.tensor(bert_tokenizer.convert_tokens_to_ids(tokens))
        token_idx = torch.tensor(token_idx.tolist() + (128 - len(token_idx)) * [0])
        return token_idx.unsqueeze(0).to(device)

    def single_bert2(message):
        tokens = bert_tokenizer.tokenize(message)
        if len(tokens) == 0:
            return []
        if tokens[0] != CLS:
            tokens = [CLS] + tokens
        if tokens[-1] != SEP:
            tokens.append(SEP)
        token_idx = torch.tensor(bert_tokenizer.convert_tokens_to_ids(tokens))
        token_idx = torch.tensor(token_idx.tolist() + (128 - len(token_idx)) * [0])
        return token_idx.unsqueeze(0).to(device)

    def get_classids(target_names, class_num):
        classids_tensor = torch.zeros(class_num, 128)
        for i in range(class_num):
            classids_tensor[i] = single_bert1(i)
        return classids_tensor.long().to(device)

    def transform_data(a_dataset):
        train_X = torch.zeros(len(a_dataset), 3, 128)
        index_X = torch.zeros(len(a_dataset))
        ground_X = torch.zeros(len(a_dataset), 128)
        words = []
        for i in tqdm(range(len(a_dataset))):
            token_idx_i, segment_idx_i, mask_i, index_i = single_bert(a_dataset[i][0])
            train_X[i] = torch.cat((token_idx_i, segment_idx_i, mask_i), dim=0)
            index_X[i] = index_i
            # prompt = get_prompt(a_dataset[i][1])
            words.append(a_dataset[i][2])
            ground_X[i] = single_bert1(a_dataset[i][1])
        train_Y = torch.tensor([i[1] for i in a_dataset]).long()
        return train_X.long(), train_Y, index_X.long(), ground_X.long(), words

    def transform_data1(a_dataset):
        train_X = torch.zeros(len(a_dataset), 3, 128)
        index_X = torch.zeros(len(a_dataset))
        words = []
        for i in tqdm(range(len(a_dataset))):
            token_idx_i, segment_idx_i, mask_i, index_i = single_bert(a_dataset[i][0])
            train_X[i] = torch.cat((token_idx_i, segment_idx_i, mask_i), dim=0)
            index_X[i] = index_i
            words.append(a_dataset[i][2])
        train_Y = torch.tensor([i[1] for i in a_dataset]).long()
        return train_X.long(), train_Y, index_X.long(), words

    def transform_data2(a_dataset):
        train_X = torch.zeros(len(a_dataset), 3, 128)
        index_X = torch.zeros(len(a_dataset))
        ground_X = torch.zeros(len(a_dataset), 128)
        words = []
        for i in tqdm(range(len(a_dataset))):
            token_idx_i, segment_idx_i, mask_i, index_i = single_bert(a_dataset[i][0])
            train_X[i] = torch.cat((token_idx_i, segment_idx_i, mask_i), dim=0)
            index_X[i] = index_i
            ground_X[i] = single_bert2(a_dataset[i][2])
            words.append(a_dataset[i][2])
        train_Y = torch.tensor([i[1] for i in a_dataset]).long()
        return train_X.long(), train_Y, index_X.long(), ground_X.long(), words

    def simcse_unloss(y_pred, device, temp=0.05):  # Cons
        y_true = torch.arange(y_pred.shape[0], device=device)
        y_true = (y_true - y_true % 2 * 2) + 1
        sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
        sim = sim - torch.eye(y_pred.shape[0], device=device) * 1e12
        sim = sim / temp
        loss = poly1_focal_loss_torch(
            sim, y_true, alpha=0.25, gamma=2, num_classes=sim.shape[0], epsilon=1.0
        )
        return torch.mean(loss), temp

    def get_img_features(words):
        img_features = []
        for word in words:
            try:
                img_features.append(img_dict[word].squeeze())
            except:
                img_features.append(torch.zeros(768))
        return torch.stack(img_features)

    def get_audio_features(words):
        audio_features = []
        for word in words:
            try:
                audio_features.append(audio_dict[word].squeeze())
            except:
                # If the word is not in the audio_dict, append a tensor of zeros
                audio_features.append(torch.zeros(768))
        # Use torch.stack to concatenate the tensors in a new dimension
        return torch.stack(audio_features)

    PAD, MASK, CLS, SEP = "[PAD]", "[MASK]", "[CLS]", "[SEP]"
    train_X, train_Y, train_index, train_grond, train_words = transform_data(train_data)
    test_X, test_Y, test_index, test_words = transform_data1(test_data)
    final_test_X, final_test_Y, final_test_index, final_euph, final_test_words = (
        transform_data2(final_test_data)
        if final_test_data_t == []
        else final_test_data_t
    )
    N_EPOCHS = 50
    num_class = 2 if IsPre else max(target_name.values()) + 1
    classids = get_classids(target_name, num_class)  # tagert/label cls
    model = Bert_In_Cross_Cons.from_pretrained(
        pretrained_model_name_or_path=BERT_model_pth,
        config=model_config,
        class_num=num_class,
        temperature=0.07,
    ).to(device)
    learning_rate = lr
    if tgt == "drug":
        N_EPOCHS = 50
    criterion = torch.nn.CrossEntropyLoss()
    criterion_img = torch.nn.CrossEntropyLoss()
    criterion_audio = torch.nn.CrossEntropyLoss()
    BATCH_SIZE = bs
    para_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in para_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in para_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_parameters, lr=learning_rate)
    num_train_steps = int(len(train_Y) / BATCH_SIZE * N_EPOCHS)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=1000,
        num_training_steps=num_train_steps,
    )

    print("[utils.py] Model Training...")
    for epoch in range(N_EPOCHS):
        since_epoch = time.time()
        model.train()
        train_loss = 0.0
        output = []
        true_labels = []
        for i in range(0, len(train_Y), BATCH_SIZE):
            batch_X = train_X[i : i + BATCH_SIZE].to(device)
            batch_Y = train_Y[i : i + BATCH_SIZE].to(device)
            batch_index = train_index[i : i + BATCH_SIZE].to(device)
            batch_size = len(batch_Y)
            batch_ground = train_grond[i : i + BATCH_SIZE].to(device)
            batch_euph = final_euph[i : i + batch_size].to(device)
            batch_X1 = final_test_X[i : i + batch_size].to(device)
            batch_index1 = final_test_index[i : i + batch_size].to(device)
            batch_words = train_words[i : i + batch_size]
            batch_img_features = get_img_features(batch_words).to(device)
            batch_audio_features = get_audio_features(batch_words).to(device)
            # out_X, matrix_Cons, matrix_Cons1, Inner_matrix, out_X1
            fusion = 1
            contrast = 0
            loss = None
            # if epoch < N_EPOCHS :
            #     fusion = 0
            if epoch < N_EPOCHS:
                contrast = 1
                out_X, out_img, out_audio, loss_audio, loss_img = model(
                    batch_X[:, 0, :],
                    batch_X[:, 1, :],
                    batch_X[:, 2, :],
                    batch_index,
                    batch_size,
                    classids,
                    fusion,
                    contrast,
                    1,
                    batch_img_features,
                    batch_audio_features,
                )
                loss_Tune = criterion(out_X, batch_Y)
                loss_img = criterion_img(out_img, batch_Y)
                loss_audio = criterion_audio(out_audio, batch_Y)
                loss = (
                    model.alpha * loss_Tune
                    + model.beta * loss_audio
                    + model.gamma * loss_img
                    + model.delta * loss_img
                    + model.epsilon * loss_audio
                )
            else:
                out_X = model(
                    batch_X[:, 0, :],
                    batch_X[:, 1, :],
                    batch_X[:, 2, :],
                    batch_index,
                    batch_size,
                    classids,
                    fusion,
                    contrast,
                    1,
                    batch_img_features,
                    batch_audio_features,
                )
                loss_Tune = criterion(out_X, batch_Y)
                loss = loss_Tune
            # Inner_loss, _ = simcse_unloss(Inner_matrix, device)

            # unlabled_loss = entropy_loss(out_X1)
            # loss_Cons, temp = simcse_unloss(matrix_Cons, device)
            # loss_Cons1, _ = simcse_unloss(matrix_Cons1, device)
            if loss is not None:
                train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            outputs_train = torch.softmax(out_X, dim=1).cpu().detach().numpy()
            acc_train = (
                100
                * sum(np.argmax(outputs_train, axis=1) == batch_Y.tolist())
                / float(len(batch_Y))
            )
            output.extend(outputs_train.tolist())
            true_labels.extend(batch_Y.tolist())

        if (epoch + 1) % 5 == 0 and (epoch + 1) != 50:
            torch.save(
                model.state_dict(),
                f"./model/model_MML_{tgt}_{epoch+1}_latest.pt",
            )
        epoch_train_acc = (
            100
            * sum(np.argmax(output, axis=1) == true_labels)
            / float(len(true_labels))
        )

        model.eval()
        _, test_acc = eval_fn1(
            test_X,
            test_Y,
            test_index,
            model,
            BATCH_SIZE,
            classids,
            confusion=0,
            msg="Testing",
            img_dict=img_dict,
            audio_dict=audio_dict,
            words=test_words,
        )
        epoch_time = (time.time() - since_epoch) / 60
        print(
            f"Epoch: {epoch + 1:02}, Train Loss: {train_loss:.3f}, Train Acc: {epoch_train_acc:.2f}%, Test Acc: {test_acc:.2f}%, using: {epoch_time:.2f}mins"
        )
    torch.save(
        model.state_dict(),
        f"./model/model_MML_{tgt}_latest.pt",
    )
    final_test_output, _ = eval_fn1(
        final_test_X,
        final_test_Y,
        final_test_index,
        model,
        BATCH_SIZE,
        classids,
        confusion=0,
        img_dict=img_dict,
        audio_dict=audio_dict,
        words=final_test_words,
    )
    final_out = convert_final_test_output_to_final_out(
        final_test_output,
        target_name,
        final_test,
        final_test_Y,
        IsPre,
        has_coarse,
        final_test_output_pre,
    )

    return model, final_out, final_test_output, (final_test_X, final_test_Y)


""" Logistic Regression """


def train_LRT_classifier(
    train_data,
    test_data,
    final_test_data,
    final_test,
    unique_vocab_dict,
    unique_vocab_list,
    target_name,
    IsPre,
    has_coarse=0,
    final_test_output_pre=[],
    final_test_data_t=[],
    bs=32,
):
    def transform_data(a_dataset, unique_vocab_dict, NGRAMS):
        train_X = torch.zeros(len(a_dataset), len(unique_vocab_dict))
        for i in tqdm(range(len(a_dataset))):
            tokens = nltk.word_tokenize(a_dataset[i][0])
            tokens = tokens if NGRAMS == 1 else ngrams_iterator(tokens, NGRAMS)
            for j in tokens:
                if j in [
                    string.punctuation,
                    "to",
                    "and",
                    "the",
                    "be",
                    "a",
                    "is",
                    "that",
                    "of",
                ]:
                    continue
                try:
                    train_X[i][unique_vocab_dict[j]] += 0.5
                except:
                    pass
        train_Y = torch.Tensor([i[1] for i in a_dataset]).long()
        return train_X, train_Y

    print("[utils.py] Transforming datasets...")
    train_X, train_Y = transform_data(train_data, unique_vocab_dict, NGRAMS=1)
    test_X, test_Y = transform_data(test_data, unique_vocab_dict, NGRAMS=1)
    final_test_X, final_test_Y = (
        transform_data(final_test_data, unique_vocab_dict, NGRAMS=1)
        if final_test_data_t == []
        else final_test_data_t
    )
    N_EPOCHS = 50
    num_class = 2 if IsPre else max(target_name.values()) + 1
    model = LR(unique_vocab_dict, unique_vocab_list, num_class=num_class).to(device)
    learning_rate = 5.0
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
    BATCH_SIZE = bs

    print("[utils.py] Model Training...")
    for epoch in range(N_EPOCHS):
        model.train()
        train_loss = 0.0
        for i in range(0, len(train_X), BATCH_SIZE):
            try:
                batch_X = train_X[i : i + BATCH_SIZE].to(device)
                batch_Y = train_Y[i : i + BATCH_SIZE].to(device)
                out_X = model(batch_X)
            except:
                batch_X = train_X[i : i + BATCH_SIZE].to(device).long()
                batch_Y = train_Y[i : i + BATCH_SIZE].to(device)
                out_X = model(batch_X)
            loss = criterion(out_X, batch_Y)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        _, train_acc = _test_result(
            train_X, train_Y, model, BATCH_SIZE, confusion=0, msg="Training"
        )
        _, test_acc = _test_result(
            test_X, test_Y, model, BATCH_SIZE, confusion=0, msg="Testing"
        )
        print(
            f"Epoch: {epoch + 1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%"
        )
    final_test_output, _ = _test_result(
        final_test_X, final_test_Y, model, BATCH_SIZE, confusion=0
    )
    final_out = convert_final_test_output_to_final_out(
        final_test_output,
        target_name,
        final_test,
        final_test_Y,
        IsPre,
        has_coarse,
        final_test_output_pre,
    )
    return model, final_out, final_test_output, (final_test_X, final_test_Y)


def eval_fn(test_X, test_Y, test_index, model, BATCH_SIZE, confusion=0, msg=""):
    output = []
    with torch.no_grad():
        for i in range(0, len(test_X), BATCH_SIZE):
            try:
                batch_X = test_X[i : i + BATCH_SIZE].to(device)
                batch_Y = test_Y[i : i + BATCH_SIZE].to(device)
                batch_index = test_index[i : i + BATCH_SIZE].to(device)
                batch_size = len(batch_Y)
                out_X = model(
                    batch_X[:, 0, :],
                    batch_X[:, 1, :],
                    batch_X[:, 2, :],
                    batch_index,
                    batch_size,
                )
            except:
                batch_X = test_X[i : i + BATCH_SIZE].to(device).long()
                batch_Y = test_Y[i : i + BATCH_SIZE].to(device)
                batch_index = test_index[i : i + BATCH_SIZE].to(device)
                batch_size = len(batch_Y)
                out_X = model(
                    batch_X[:, 0, :],
                    batch_X[:, 1, :],
                    batch_X[:, 2, :],
                    batch_index,
                    batch_size,
                )
            output.extend(out_X.tolist())
        acc = (
            100 * sum(np.argmax(output, axis=1) == test_Y.tolist()) / float(len(test_Y))
        )

        if confusion == 1:
            GT = test_Y.tolist()
            ours = np.array(output).argmax(1).tolist()
            print(msg, end=" ")
            print(f"Accuracy: {acc:.2f}%")
            print(confusion_matrix(GT, ours))

        return output, acc


def eval_fn1(
    test_X,
    test_Y,
    test_index,
    model,
    BATCH_SIZE,
    classids,
    confusion=0,
    msg="",
    img_dict={},
    audio_dict={},
    words=[],  # For img and audio
):
    def get_img_features(words):
        img_features = []
        for word in words:
            try:
                img_features.append(img_dict[word].squeeze())
            except:
                img_features.append(torch.zeros(768))
        return torch.stack(img_features)

    def get_audio_features(words):
        audio_features = []
        for word in words:
            try:
                # Use .squeeze(0) to remove the first dimension
                audio_features.append(audio_dict[word].squeeze())
            except:
                # If the word is not in the audio_dict, append a tensor of zeros
                audio_features.append(torch.zeros(768))
        # Use torch.stack to concatenate the tensors in a new dimension
        return torch.stack(audio_features)

    output = []
    with torch.no_grad():
        for i in range(0, len(test_X), BATCH_SIZE):
            batch_X = test_X[i : i + BATCH_SIZE].to(device)
            batch_Y = test_Y[i : i + BATCH_SIZE].to(device)
            batch_index = test_index[i : i + BATCH_SIZE].to(device)
            batch_size = len(batch_Y)
            batch_words = words[i : i + batch_size]
            batch_img_features = get_img_features(batch_words).to(device)
            batch_audio_features = get_audio_features(batch_words).to(device)
            out_X = model(
                input_ids=batch_X[:, 0, :],
                token_type_ids=batch_X[:, 1, :],
                input_mask=batch_X[:, 2, :],
                positions=batch_index,
                batch_size=batch_size,
                class_cls=classids,
                fusion=1,
                img_ids=batch_img_features,
                audio_ids=batch_audio_features,
            )
            output.extend(out_X.tolist())
        acc = (
            100 * sum(np.argmax(output, axis=1) == test_Y.tolist()) / float(len(test_Y))
        )

        if confusion == 1:
            GT = test_Y.tolist()
            ours = np.array(output).argmax(1).tolist()
            print(msg, end=" ")
            print(f"Accuracy: {acc:.2f}%")
            print(confusion_matrix(GT, ours))

        return output, acc


def _test_result(test_X, test_Y, model, BATCH_SIZE, confusion=0, msg=""):
    output = []
    with torch.no_grad():
        for i in range(0, len(test_X), BATCH_SIZE):
            try:
                batch_X = test_X[i : i + BATCH_SIZE].to(device)
                out_X = model(batch_X)
            except:
                batch_X = test_X[i : i + BATCH_SIZE].to(device).long()
                out_X = model(batch_X)
            output.extend(out_X.tolist())

    acc = 100 * sum(np.argmax(output, 1) == test_Y.tolist()) / float(len(test_Y))
    if confusion == 1:
        GT = test_Y.tolist()
        ours = np.array(output).argmax(1).tolist()
        print(msg, end=" ")
        print(f"Accuracy: {acc:.2f}%")
        print(confusion_matrix(GT, ours))
    return output, acc


""" Neural Models """


def train_initialization(
    classifier_name, train_data, test_data, Final_test, target_name, bs, IsPre=0
):
    def load_dataset(train_data, test_data, Final_test, embedding_length, batch_size):
        """
        tokenizer : Breaks sentences into a list of words. If sequential=False, no tokenization is applied
        Field : A class that stores information about the way of preprocessing
        fix_length : An important property of TorchText is that we can let the input to be variable length, and TorchText will
                     dynamically pad each sequence to the longest sequence in that "batch". But here we are using fi_length which
                     will pad each sequence to have a fix length of 200.

        build_vocab : It will first make a vocabulary or dictionary mapping all the unique words present in the train_data to an
                      idx and then after it will use GloVe word embedding to map the index to the corresponding word embedding.

        vocab.vectors : This returns a torch tensor of shape (vocab_size x embedding_dim) containing the pre-trained word embeddings.
        BucketIterator : Defines an iterator that batches examples of similar lengths together to minimize the amount of padding needed.

        """

        def get_dataset(a_data, fields):
            examples = []
            for data_i in tqdm(a_data):
                examples.append(
                    torchtext.data.Example.fromlist([data_i[0], data_i[1]], fields)
                )
            return examples

        tokenize = lambda x: x.split()
        TEXT = torchtext.data.Field(
            sequential=True,
            tokenize=tokenize,
            lower=True,
            include_lengths=True,
            batch_first=True,
            fix_length=15,
        )
        LABEL = torchtext.data.LabelField()
        fields = [("text", TEXT), ("label", LABEL)]
        train_data = get_dataset(train_data, fields)
        test_data = get_dataset(test_data, fields)
        Final_test = get_dataset(Final_test, fields)
        train_data = torchtext.data.Dataset(train_data, fields=fields)
        test_data = torchtext.data.Dataset(test_data, fields=fields)
        Final_test = torchtext.data.Dataset(Final_test, fields=fields)

        TEXT.build_vocab(train_data, vectors=GloVe(name="6B", dim=embedding_length))
        LABEL.build_vocab(train_data)
        word_embeddings = TEXT.vocab.vectors
        vocab_size = len(TEXT.vocab)
        print("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
        print("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
        print("Label Length: " + str(len(LABEL.vocab)))
        train_iter, test_iter, Final_test_iter = torchtext.data.Iterator.splits(
            (train_data, test_data, Final_test),
            batch_size=batch_size,
            sort=False,
            repeat=False,
        )
        return TEXT, vocab_size, word_embeddings, train_iter, test_iter, Final_test_iter

    output_size = 2 if IsPre else max(target_name.values()) + 1
    learning_rate = 0.002
    hidden_size = 256
    embedding_length = 100
    epoch_num = 3 if IsPre else 10
    batch_size = bs
    pre_train = True
    embedding_tune = False

    (
        TEXT,
        vocab_size,
        word_embeddings,
        train_iter,
        test_iter,
        Final_test_iter,
    ) = load_dataset(
        train_data,
        test_data,
        [[x[0], 0] for x in Final_test],
        embedding_length,
        batch_size,
    )
    if classifier_name == "LSTM":
        model = LSTM(
            batch_size,
            output_size,
            hidden_size,
            vocab_size,
            embedding_length,
            word_embeddings,
            pre_train,
            embedding_tune,
        )
    elif classifier_name == "CNN":
        model = CNN(
            batch_size,
            output_size,
            1,
            64,
            [2, 3, 4],
            2,
            0,
            0.2,
            vocab_size,
            embedding_length,
            word_embeddings,
            pre_train,
            embedding_tune,
        )
    elif classifier_name == "LSTMAtten":
        model = LSTM_AttentionModel(
            batch_size,
            output_size,
            hidden_size,
            vocab_size,
            embedding_length,
            word_embeddings,
            pre_train,
            embedding_tune,
        )
    else:
        raise ValueError("Not a valid classifier_name!!!")
    model = model.to(device)
    loss_fn = F.cross_entropy
    return (
        train_iter,
        test_iter,
        Final_test_iter,
        learning_rate,
        epoch_num,
        model,
        loss_fn,
    )


def train_model(model, train_iter, loss_fn, learning_rate, epoch, bs):
    def clip_gradient(model, clip_value):
        params = list(filter(lambda p: p.grad is not None, model.parameters()))
        for p in params:
            p.grad.data.clamp_(-clip_value, clip_value)

    total_epoch_loss = 0
    total_epoch_acc = 0
    optim = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate
    )
    steps = 0
    model.train()
    for idx, batch in enumerate(train_iter):
        text = batch.text[0]
        target = batch.label
        target = torch.autograd.Variable(target).long()
        text = text.to(device)
        target = target.to(device)
        if text.size()[0] != bs:
            continue
        optim.zero_grad()
        prediction = model(text)
        loss = loss_fn(prediction, target)
        num_corrects = (
            (torch.max(prediction, 1)[1].view(target.size()).data == target.data)
            .float()
            .sum()
        )
        acc = 100.0 * num_corrects / len(batch)
        loss.backward()
        clip_gradient(model, 1e-1)
        optim.step()
        steps += 1

        if steps % 1000 == 0:
            print(
                f"Epoch: {epoch + 1}, Idx: {idx + 1}, Training Loss: {loss.item():.4f}, Training Accuracy: {acc.item(): .2f}%"
            )

        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()

    return total_epoch_loss / len(train_iter), total_epoch_acc / len(train_iter)


def eval_model(model, val_iter, loss_fn, bs):
    total_epoch_loss = 0
    total_epoch_acc = 0
    all_prediction = []
    all_target = []
    model.eval()
    with torch.no_grad():
        try:
            for idx, batch in enumerate(val_iter):
                text = batch.text[0]
                if text.size()[0] != bs:
                    continue
                target = batch.label
                target = torch.autograd.Variable(target).long()
                text = text.to(device)
                target = target.to(device)
                prediction = model(text)
                all_prediction.extend(prediction.tolist())
                all_target.extend(target.tolist())
                loss = loss_fn(prediction, target)
                num_corrects = (
                    torch.max(prediction, 1)[1].view(target.size()).data == target.data
                ).sum()
                acc = 100.0 * num_corrects / len(batch)
                total_epoch_loss += loss.item()
                total_epoch_acc += acc.item()
        except:
            pass
    return (
        total_epoch_loss / len(val_iter),
        total_epoch_acc / len(val_iter),
        all_prediction,
    )


def get_train_test_data1(
    input_keywords, target_name, all_text, final_test, NGRAMS, train_perc
):
    print("[utils.py] Constructing train and test data...")
    all_data = []
    all_data_pre = []
    final_test_data = []
    for i in tqdm(all_text):
        temp = nltk.word_tokenize(i)
        for j, keyword in enumerate(
            input_keywords
        ):  # Add positive labels that belong to input keywords.
            if keyword not in temp:
                continue
            temp_index = temp.index(keyword)
            masked_sentence = (
                " ".join(temp[:temp_index])
                + " [MASK] "
                + " ".join(temp[temp_index + 1 :])
            )
            all_data.append([masked_sentence, target_name[keyword], keyword])
            all_data_pre.append([masked_sentence, 1])  # is one of the target keywords.
        temp_index = random.randint(0, len(temp) - 1)
        if (
            temp[temp_index] not in input_keywords
        ):  # Add negative labels that NOT belong to input keywords.
            masked_sentence = (
                " ".join(temp[:temp_index])
                + " [MASK] "
                + " ".join(temp[temp_index + 1 :])
            )
            all_data_pre.append(
                [masked_sentence, 0]
            )  # is NOT one of the target keywords.
        for j, keyword in enumerate(final_test):  # Construct final_test_data
            if keyword not in temp:
                continue
            temp_index = temp.index(keyword)
            masked_sentence = (
                " ".join(temp[:temp_index])
                + " [MASK] "
                + " ".join(temp[temp_index + 1 :])
            )
            final_test_data.append([masked_sentence, j, keyword])

    def _shuffle_and_balance(all_data, max_len):
        random.shuffle(all_data)
        data_len = defaultdict(int)
        all_data_balanced = []
        for i in all_data:
            if data_len[i[1]] == max_len:
                continue
            data_len[i[1]] += 1
            all_data_balanced.append(i)
        random.shuffle(all_data_balanced)
        train_data = all_data_balanced[: int(train_perc * len(all_data_balanced))]
        test_data = all_data_balanced[int(train_perc * len(all_data_balanced)) :]
        return train_data, test_data

    train_data, test_data = _shuffle_and_balance(all_data, max_len=2000)
    train_data_pre, test_data_pre = _shuffle_and_balance(
        all_data_pre,
        max_len=min(
            100000,
            sum([x[1] == 0 for x in all_data_pre]),
            sum([x[1] == 1 for x in all_data_pre]),
        ),
    )
    unique_vocab_dict, unique_vocab_list = build_vocab(train_data, NGRAMS, min_count=10)
    return (
        train_data,
        test_data,
        final_test_data,
        train_data_pre,
        test_data_pre,
        unique_vocab_dict,
        unique_vocab_list,
    )


""" Utility Functions """


def get_filtered_final_out(final_out, final_test, input_keywords, target_name, tgt):
    print("\n" + "-" * 40 + " [Final Results] " + "-" * 40)
    final_top_words = []
    filtered_final_out = []
    filtered_final_test = {}
    for i, word in enumerate(final_test):
        if final_out[i] == [len(input_keywords)]:
            continue
        final_top_words.append(word)
        if final_test[word] != ["None"]:
            filtered_final_out.append(final_out[i])
            filtered_final_test[word] = final_test[word]
    print_final_out(filtered_final_out, filtered_final_test, target_name, tgt)
    return 0


def print_final_out(final_out, final_test, target_name, tgt):
    ranking_list = []
    target_name_list = []
    for i in range(max(target_name.values()) + 1):
        target_name_list.append([x for x in target_name if target_name[x] == i])
    for i, word in enumerate(final_test):
        # print('{:12s}: \t'.format(word), end='')
        position = 0
        for j in final_out[i]:
            position += 1
            if any(ele in target_name_list[j] for ele in final_test[word]):
                break
        ranking_list.append(position)
    print(
        "Average ranking is {:.2f} for {:d} euphemisms.".format(
            sum(ranking_list) / len(ranking_list), len(ranking_list)
        )
    )
    topk_acc = [
        sum(x <= k + 1 for x in ranking_list) / len(final_test)
        for k in range(len(target_name_list))
    ]
    print("[Top-k Accuracy]: ", end="")
    for k in range(len(target_name_list)):
        print("|  {:2d}  ".format(k + 1), end="")
    print()
    print(" " * 18, end="")
    # ----
    file = open(f"statistics_{tgt}.csv")
    reader = csv.reader(file)
    original = list(reader)
    file.close()
    statistics = open(f"statistics_{tgt}.csv", "w", encoding="utf-8")
    writer = csv.writer(statistics, delimiter=",")
    for row in original:
        writer.writerow(row)
    csvrow = []
    # ----

    for k in range(len(target_name_list)):
        print("| {:.2f} ".format(topk_acc[k]), end="")
        # ---
        if len(csvrow) <= 2:
            csvrow.append("{:.2f}".format(topk_acc[k]))
    writer.writerow(csvrow)
    statistics.close()
    # ---
    print()
    return 0


def convert_final_test_output_to_final_out(
    final_test_output,
    target_name,
    final_test,
    final_test_Y,
    IsPre,
    has_coarse,
    final_test_output_pre=[],
):
    final_test_output = softmax(final_test_output, axis=1).tolist()
    if IsPre:
        final_out = [np.array([0.0, 0.0]) for x in range(len(final_test))]
        for i, j in enumerate(final_test_output):
            if j[0] < j[1]:
                final_out[final_test_Y[i]] += j
        return final_out

    final_out = [
        np.array([0.0 for y in range(max(target_name.values()) + 1)])
        for x in range(len(final_test))
    ]
    for i, j in enumerate(final_test_output):
        if has_coarse and i < len(final_test_output_pre):
            if final_test_output_pre[i][0] < final_test_output_pre[i][1]:
                final_out[final_test_Y[i]] += j
        else:
            final_out[final_test_Y[i]] += j
    for i in range(len(final_out)):
        if sum(final_out[i]) == 0:
            final_out[i] = [len(target_name)]
        else:
            final_out[i] = np.argsort(final_out[i])[::-1].tolist()
    return final_out


def get_final_test(euphemism_answer, top_words, input_keywords):
    final_test = {}
    for x in top_words:
        if x in euphemism_answer:
            if any(ele in euphemism_answer[x] for ele in input_keywords):
                final_test[x] = euphemism_answer[x]
            else:
                final_test[x] = ["None"]
        else:
            final_test[x] = ["None"]
    return final_test


def build_vocab(xlist, NGRAMS, min_count):
    vocabi2w = ["[SOS]", "[EOS]", "[PAD]", "[UNK]"]  # A list of unique words
    seen = defaultdict(int)
    for i in range(len(xlist)):
        tokens = nltk.word_tokenize(xlist[i][0])
        tokens = tokens if NGRAMS == 1 else ngrams_iterator(tokens, NGRAMS)
        for token in tokens:
            seen[token] += 1
    vocabi2w += [x for x in seen if seen[x] >= min_count]
    vocabw2i = {vocabi2w[x]: x for x in range(len(vocabi2w))}
    return vocabw2i, vocabi2w
