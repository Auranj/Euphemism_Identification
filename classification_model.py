import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from random import sample
from transformers import BertTokenizer, get_linear_schedule_with_warmup
import torch.nn.init as init
from transformers import BertModel, BertPreTrainedModel
import math
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_tokenizer = BertTokenizer(vocab_file="data/BERT_model_reddit/vocab.txt")


class CrossAttention(nn.Module):
    def __init__(self, feature_dim):
        super(CrossAttention, self).__init__()
        self.text_linear = nn.Linear(feature_dim, feature_dim)
        self.extra_linear = nn.Linear(feature_dim, feature_dim)
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, query, key, value):
        if query.shape[-1] != 768:
            query = self.text_linear(query)  # [batch, seq_len, feature_dim]
        if key.shape[-1] != 768:
            key = self.extra_linear(key)
            value = self.extra_linear(value)
        query = self.query_proj(query)  # [batch, feature_dim]
        key = self.key_proj(key)  # [batch, feature_dim]
        value = self.value_proj(value)  # [batch, feature_dim]
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / torch.sqrt(
            torch.tensor(key.size(-1), dtype=torch.float32)
        )
        attention_weights = F.softmax(attention_scores, dim=-1)
        attended_values = torch.matmul(attention_weights, value)
        return attended_values


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        return x


class GatedUnit(nn.Module):
    def __init__(self, input_dim):
        super(GatedUnit, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.gate(x)


class AddNorm(nn.Module):
    def __init__(self, feature_dim):
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x, y):
        return self.norm(x + y)


class SelfAttention(nn.Module):
    def __init__(self, feature_dim, dropout_prob=0.1):
        super(SelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(feature_dim, num_heads=1)

    def forward(self, x):
        return self.attention(x, x, x)[0]


class AdvancedGatedFusion(nn.Module):
    def __init__(self, feature_dim):
        super(AdvancedGatedFusion, self).__init__()
        self.feature_dim = feature_dim

        # 定义更复杂的门控网络
        self.img_gate = nn.Sequential(
            nn.Linear(2 * feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Tanh(),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid(),
        )
        self.audio_gate = nn.Sequential(
            nn.Linear(2 * feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Tanh(),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(2 * feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )

        # 定义图像和音频的独立权重
        self.image_weight = nn.Parameter(torch.rand(feature_dim))
        self.audio_weight = nn.Parameter(torch.rand(feature_dim))

    def forward(self, text_features, image_features, audio_features):
        # 拼接三种模态的特征
        combined_img_text_features = torch.cat((text_features, image_features), dim=1)
        combined_audio_text_features = torch.cat((text_features, audio_features), dim=1)
        # 计算门控信号
        img_gate_signal = self.img_gate(combined_img_text_features)
        audio_gate_signal = self.audio_gate(combined_audio_text_features)
        # 计算加权融合特征
        weighted_image = image_features * img_gate_signal
        weighted_audio = audio_features * audio_gate_signal
        fused_features = torch.cat((weighted_image, weighted_audio), dim=1)
        fused_features = self.mlp(fused_features)

        return fused_features


class Bert_In_Cross_Cons(BertPreTrainedModel):
    def __init__(self, config, class_num, temperature=None):
        super(Bert_In_Cross_Cons, self).__init__(config)
        self.bert = BertModel(config=config)
        self.text_extractor = ProjectionHead(config.hidden_size, config.hidden_size)
        self.img_extractor = ProjectionHead(config.hidden_size, config.hidden_size)
        self.audio_extractor = ProjectionHead(config.hidden_size, config.hidden_size)
        self.idiom_embedding = nn.Embedding(class_num, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.classifier_img = nn.Linear(config.hidden_size, 1)
        self.classifier_audio = nn.Linear(config.hidden_size, 1)
        self.class_num = class_num
        self.temperature = temperature
        self.alpha = torch.rand(1, requires_grad=True).to(device)
        self.beta = torch.rand(1, requires_grad=True).to(device)
        self.gamma = torch.rand(1, requires_grad=True).to(device)
        self.delta = torch.rand(1, requires_grad=True).to(device)
        self.epsilon = torch.rand(1, requires_grad=True).to(device)
        self.cross_attention1 = CrossAttention(config.hidden_size)
        self.cross_attention2 = CrossAttention(config.hidden_size)
        self.add_norm_layer1 = AddNorm(config.hidden_size)
        self.add_norm_layer2 = AddNorm(config.hidden_size)
        self.self_attention_layer_img = SelfAttention(
            config.hidden_size, config.hidden_dropout_prob
        )
        self.self_attention_layer_audio = SelfAttention(
            config.hidden_size, config.hidden_dropout_prob
        )
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
        )
        self.ff = AdvancedGatedFusion(config.hidden_size)
        self.img_gate = GatedUnit(config.hidden_size)
        self.audio_gate = GatedUnit(config.hidden_size)
        self.init_weights()

    def infoNCE_loss(self, features1, features2, temperature=0.07):
        """
        计算InfoNCE损失
        :param features1: 第一组特征向量，维度为 [batch_size, feature_dim]
        :param features2: 第二组特征向量，维度为 [batch_size, feature_dim]
        :param temperature: 温度参数，控制软化程度
        :return: InfoNCE损失
        """
        # 归一化特征向量
        features1 = F.normalize(features1, p=2, dim=1)
        features2 = F.normalize(features2, p=2, dim=1)

        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features1, features2.T) / temperature

        # 对角线元素是正样本对的相似度
        labels = torch.arange(features1.size(0)).to(features1.device)
        loss = F.cross_entropy(similarity_matrix, labels)

        return loss

    def forward(
        self,
        input_ids,
        token_type_ids,
        input_mask,
        positions,
        batch_size,
        class_cls=None,
        fusion=0,
        contrast=0,
        Is_train=0,
        img_ids=None,
        audio_ids=None,
    ):
        # sequence_outputs, cls_outputs = self.bert(input_ids, input_mask, token_type_ids)
        sequence_output, cls_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
        )[
            :2
        ]  # [batch,128,768] [batch,768]
        # print("sequence_output:",sequence_output.shape)
        key_word_context = sequence_output[torch.arange(batch_size), positions]
        # print("context:",key_word_context.shape)
        blank_states = self.text_extractor(cls_outputs)  # [batch,768]
        if img_ids is not None:
            audio_ids = self.audio_extractor(audio_ids)
            img_ids = self.img_extractor(img_ids)
            loss_audio = self.infoNCE_loss(blank_states, audio_ids, self.temperature)
            loss_img = self.infoNCE_loss(blank_states, img_ids, self.temperature)
            if fusion != 0:
                image_attended = self.cross_attention1(blank_states, img_ids, img_ids)
                audio_attended = self.cross_attention2(
                    blank_states, audio_ids, audio_ids
                )
                image_gate_weights = self.img_gate(image_attended)
                audio_gate_weights = self.audio_gate(audio_attended)
                image_attended = image_gate_weights * image_attended
                audio_attended = audio_gate_weights * audio_attended
                add_norm_img = self.add_norm_layer1(blank_states, image_attended)
                add_norm_audio = self.add_norm_layer1(blank_states, audio_attended)

                # Self-attention for Image and Audio
                self_attended_img = self.self_attention_layer_img(add_norm_img)
                self_attended_audio = self.self_attention_layer_audio(add_norm_audio)

                # Second Add & Norm for Image and Audio
                final_img_features = self.add_norm_layer2(
                    add_norm_img, self_attended_img
                )
                final_audio_features = self.add_norm_layer2(
                    add_norm_audio, self_attended_audio
                )

                # Concatenate and MLP
                concatenated_features = torch.cat(
                    [final_img_features, final_audio_features], dim=1
                )
                fused_features = self.mlp(concatenated_features)
                # fused_features = blank_states + image_attended + audio_attended
                # fused_features = self.ff(blank_states, img_ids, audio_ids)
                blank_states = fused_features
            # print(cls_outputs.shape,audio_ids.shape,img_ids.shape)

        class_ids = torch.zeros(batch_size, self.class_num).long().to(device)
        for i in range(batch_size):
            class_ids[i] = torch.tensor([j for j in range(self.class_num)]).long()
        encoded_idiom = self.idiom_embedding(class_ids)  # [batch, 33, hidden_state]

        multiply_result = torch.einsum(
            "abc,ac->abc", encoded_idiom, blank_states
        )  # [batch，33，hidden_state]
        pooled_output = self.dropout(multiply_result)

        logits = self.classifier(pooled_output)  # cross entropy loss [batch,33,1]
        logits = logits.view(-1, class_ids.shape[-1])  # [batch, 33]
        # print(logits.shape)
        if fusion != 0:
            # image classify loss
            multiply_result_img = torch.einsum(
                "abc,ac->abc", encoded_idiom, img_ids
            )  # [batch，33，hidden_state]
            pooled_output_img = self.dropout(multiply_result_img)
            logits_img = self.classifier_img(pooled_output_img)
            logits_img = logits.view(-1, class_ids.shape[-1])

            # audio classify loss
            multiply_result_audio = torch.einsum(
                "abc,ac->abc", encoded_idiom, audio_ids
            )
            pooled_output_audio = self.dropout(multiply_result_audio)
            logits_audio = self.classifier_audio(pooled_output_audio)
            logits_audio = logits.view(-1, class_ids.shape[-1])

        if Is_train == 0:
            return logits
        if img_ids is not None and contrast == 1:
            return logits, logits_img, logits_audio, loss_audio, loss_img
        return logits


class LR(nn.Module):  # Logsitic Regression
    def __init__(self, unique_vocab_dict, unique_vocab_list, num_class):
        super().__init__()
        self.unique_vocab_dict = unique_vocab_dict
        self.unique_vocab_list = unique_vocab_list
        self.vocab_size = len(unique_vocab_dict)
        self.num_class = num_class
        self.fc = nn.Linear(self.vocab_size, num_class)

    def forward(self, text):
        out = self.fc(text)
        return out
        # return torch.softmax(out, dim=1)

    def get_features(self, class_names, num=50):
        print("-----------------------")
        features = []
        for i in range(self.num_class):
            feature = []
            sorted_weight_index = self.fc.weight[i].argsort().tolist()[::-1]
            for j in range(min(num, len(sorted_weight_index))):
                feature.append(self.unique_vocab_list[sorted_weight_index[j]])
            print(class_names[i], end=": ")
            print(feature)
            features.append(feature)
        print("-----------------------")
        return features


class LR_embeddings(nn.Module):  # Logsitic Regression with embeddings
    def __init__(self, unique_vocab_dict, embedding_length, num_class):
        super().__init__()
        self.vocab_size = len(unique_vocab_dict)
        self.embedding = nn.EmbeddingBag(
            self.vocab_size, embedding_dim=embedding_length, sparse=True
        )
        self.fc = nn.Linear(embedding_length, num_class)

    def forward(self, text):
        out = self.embedding(text)
        out = self.fc(out)
        return torch.softmax(out, dim=1)


class LSTM_AttentionModel(nn.Module):
    def __init__(
        self,
        batch_size,
        output_size,
        hidden_size,
        vocab_size,
        embedding_length,
        weights,
        pre_train,
        embedding_tune,
    ):
        super(LSTM_AttentionModel, self).__init__()

        """
        Arguments
        ---------
        batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        output_size : 2 = (pos, neg)
        hidden_sie : Size of the hidden_state of the LSTM
        vocab_size : Size of the vocabulary containing unique words
        embedding_length : Embeddding dimension of GloVe word embeddings
        weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 

        --------

        """

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length

        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
        if pre_train:
            self.word_embeddings.weights = nn.Parameter(
                weights, requires_grad=embedding_tune
            )
        self.lstm = nn.LSTM(embedding_length, hidden_size)
        self.label = nn.Linear(hidden_size, output_size)

    # self.attn_fc_layer = nn.Linear()

    def attention_net(self, lstm_output, final_state):
        """
        Now we will incorporate Attention mechanism in our LSTM model. In this new model, we will use attention to compute soft alignment score corresponding
        between each of the hidden_state and the last hidden_state of the LSTM. We will be using torch.bmm for the batch matrix multiplication.

        Arguments
        ---------

        lstm_output : Final output of the LSTM which contains hidden layer outputs for each sequence.
        final_state : Final time-step hidden state (h_n) of the LSTM

        ---------

        Returns : It performs attention mechanism by first computing weights for each of the sequence present in lstm_output and and then finally computing the
                  new hidden state.

        Tensor Size :
                    hidden.size() = (batch_size, hidden_size)
                    attn_weights.size() = (batch_size, num_seq)
                    soft_attn_weights.size() = (batch_size, num_seq)
                    new_hidden_state.size() = (batch_size, hidden_size)

        """

        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(
            lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)
        ).squeeze(2)

        return new_hidden_state

    def forward(self, input_sentences, batch_size=None):
        """
        Parameters
        ----------
        input_sentence: input_sentence of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)

        Returns
        -------
        Output of the linear layer containing logits for pos & neg class which receives its input as the new_hidden_state which is basically the output of the Attention network.
        final_output.shape = (batch_size, output_size)

        """

        input = self.word_embeddings(input_sentences)
        input = input.permute(1, 0, 2)
        if batch_size is None:
            h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).to(device))
            c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).to(device))
        else:
            h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).to(device))
            c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).to(device))

        output, (final_hidden_state, final_cell_state) = self.lstm(
            input, (h_0, c_0)
        )  # final_hidden_state.size() = (1, batch_size, hidden_size)
        output = output.permute(
            1, 0, 2
        )  # output.size() = (batch_size, num_seq, hidden_size)

        attn_output = self.attention_net(output, final_hidden_state)
        logits = self.label(attn_output)

        return logits


class CNN(nn.Module):
    def __init__(
        self,
        batch_size,
        output_size,
        in_channels,
        out_channels,
        kernel_heights,
        stride,
        padding,
        keep_probab,
        vocab_size,
        embedding_length,
        weights,
        pre_train,
        embedding_tune,
    ):
        super(CNN, self).__init__()

        """
        Arguments
        ---------
        batch_size : Size of each batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        output_size : 2 = (pos, neg)
        in_channels : Number of input channels. Here it is 1 as the input data has dimension = (batch_size, num_seq, embedding_length)
        out_channels : Number of output channels after convolution operation performed on the input matrix
        kernel_heights : A list consisting of 3 different kernel_heights. Convolution will be performed 3 times and finally results from each kernel_height will be concatenated.
        keep_probab : Probability of retaining an activation node during dropout operation
        vocab_size : Size of the vocabulary containing unique words
        embedding_length : Embedding dimension of GloVe word embeddings
        weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table
        --------

        """
        self.batch_size = batch_size
        self.output_size = output_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_heights = kernel_heights
        self.stride = stride
        self.padding = padding
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
        if pre_train:
            self.word_embeddings.weight = nn.Parameter(
                weights, requires_grad=embedding_tune
            )
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            (kernel_heights[0], embedding_length),
            stride,
            padding,
        )
        self.conv2 = nn.Conv2d(
            in_channels,
            out_channels,
            (kernel_heights[1], embedding_length),
            stride,
            padding,
        )
        self.conv3 = nn.Conv2d(
            in_channels,
            out_channels,
            (kernel_heights[2], embedding_length),
            stride,
            padding,
        )
        self.dropout = nn.Dropout(keep_probab)
        self.label = nn.Linear(len(kernel_heights) * out_channels, output_size)
        self.conv4 = nn.Conv1d(
            len(kernel_heights) * out_channels, out_channels, 3, padding=1
        )
        self.final_dropout = nn.Dropout(keep_probab)
        self.final_label = nn.Linear(out_channels, output_size)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)

    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(
            input
        )  # conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = F.relu(
            conv_out.squeeze(3)
        )  # activation.size() = (batch_size, out_channels, dim1)
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(
            2
        )  # maxpool_out.size() = (batch_size, out_channels)
        return max_out

    def additional_conv_block(self, input, conv_layer):
        conv_out = conv_layer(
            input
        )  # conv_out.size() = (batch_size, out_channels, dim)
        activation = F.elu(conv_out)
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(
            2
        )  # maxpool_out.size() = (batch_size, out_channels)
        return max_out

    def forward(self, input_sentences, batch_size=None):
        """
        The idea of the Convolutional Neural Netwok for Text Classification is very simple. We perform convolution operation on the embedding matrix
        whose shape for each batch is (num_seq, embedding_length) with kernel of varying height but constant width which is same as the embedding_length.
        We will be using ReLU activation after the convolution operation and then for each kernel height, we will use max_pool operation on each tensor
        and will filter all the maximum activation for every channel and then we will concatenate the resulting tensors. This output is then fully connected
        to the output layers consisting two units which basically gives us the logits for both positive and negative classes.

        Parameters
        ----------
        input_sentences: input_sentences of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)

        Returns
        -------
        Output of the linear layer containing logits for pos & neg class.
        logits.size() = (batch_size, output_size)

        """

        input = self.word_embeddings(input_sentences)
        input = input.unsqueeze(1)
        max_out1 = self.conv_block(input, self.conv1)
        max_out2 = self.conv_block(input, self.conv2)
        max_out3 = self.conv_block(input, self.conv3)

        all_out = torch.cat((max_out1, max_out2, max_out3), 1)
        fc_in = self.dropout(all_out)
        fc_in_conv = self.additional_conv_block(fc_in.unsqueeze(2), self.conv4)
        logits = self.final_label(fc_in_conv)
        return logits


class LSTM(nn.Module):
    def __init__(
        self,
        batch_size,
        output_size,
        hidden_size,
        vocab_size,
        embedding_length,
        weights,
        pre_train,
        embedding_tune,
    ):
        super(LSTM, self).__init__()

        """
        Arguments
        ---------
        batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        output_size : 2 = (pos, neg)
        hidden_sie : Size of the hidden_state of the LSTM
        vocab_size : Size of the vocabulary containing unique words
        embedding_length : Embeddding dimension of GloVe word embeddings
        weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 

        """

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length

        self.word_embeddings = nn.Embedding(
            vocab_size, embedding_length
        )  # Initializing the look-up table.
        if pre_train:
            self.word_embeddings.weight = nn.Parameter(
                weights, requires_grad=embedding_tune
            )  # Assigning the look-up table to the pre-trained GloVe word embedding.
        self.lstm = nn.LSTM(embedding_length, hidden_size)
        self.label = nn.Linear(hidden_size, output_size)

    def forward(self, input_sentence, batch_size=None):
        """
        Parameters
        ----------
        input_sentence: input_sentence of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)

        Returns
        -------
        Output of the linear layer containing logits for positive & negative class which receives its input as the final_hidden_state of the LSTM
        final_output.shape = (batch_size, output_size)

        """

        """ Here we will map all the indexes present in the input sequence to the corresponding word vector using our pre-trained word_embedddins."""
        input = self.word_embeddings(
            input_sentence
        )  # embedded input of shape = (batch_size, num_sequences, embedding_length)
        input = input.permute(
            1, 0, 2
        )  # input.size() = (num_sequences, batch_size, embedding_length)
        if batch_size is None:
            h_0 = Variable(
                torch.zeros(1, self.batch_size, self.hidden_size).to(device)
            )  # Initial hidden state of the LSTM
            c_0 = Variable(
                torch.zeros(1, self.batch_size, self.hidden_size).to(device)
            )  # Initial cell state of the LSTM
        else:
            h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).to(device))
            c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).to(device))
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
        final_output = self.label(
            final_hidden_state[-1]
        )  # final_hidden_state.size() = (1, batch_size, hidden_size) & final_output.size() = (batch_size, output_size)
        return final_output
