#! -*- coding: utf-8 -*-
# 用CRF做中文命名实体识别
# 数据集 https://github.com/CLUEbenchmark/CLUENER2020

import json
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open, ViterbiDecoder, to_array
from bert4keras.layers import ConditionalRandomField
from keras.layers import Dense
from keras.models import Model
from tqdm import tqdm

maxlen = 256
epochs = 10
batch_size = 16
learning_rate = 2e-5
crf_lr_multiplier = 1000  # 必要时扩大CRF层的学习率
categories = set()

# bert配置
config_path = '/root/kg/bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/root/kg/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/root/kg/bert/chinese_L-12_H-768_A-12/vocab.txt'


def load_data(filename):
    """加载数据
    单条格式：[text, (start, end, label), (start, end, label), ...]，
              意味着text[start:end + 1]是类型为label的实体。
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            d = [l['text']]
            for k, v in l['label'].items():
                categories.add(k)
                for spans in v.values():
                    for start, end in spans:
                        d.append((start, end, k))
            D.append(d)
    return D


# 标注数据
train_data = load_data('/root/ner/cluener/train.json')
valid_data = load_data('/root/ner/cluener/dev.json')
categories = list(sorted(categories))

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, d in self.sample(random):
            tokens = tokenizer.tokenize(d[0], maxlen=maxlen)
            mapping = tokenizer.rematch(d[0], tokens)
            start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
            end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
            token_ids = tokenizer.tokens_to_ids(tokens)
            segment_ids = [0] * len(token_ids)
            labels = np.zeros(len(token_ids))
            for start, end, label in d[1:]:
                if start in start_mapping and end in end_mapping:
                    start = start_mapping[start]
                    end = end_mapping[end]
                    labels[start] = categories.index(label) * 2 + 1
                    labels[start + 1:end + 1] = categories.index(label) * 2 + 2
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


model = build_transformer_model(config_path, checkpoint_path)
output = Dense(len(categories) * 2 + 1)(model.output)
CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
output = CRF(output)

model = Model(model.input, output)
model.summary()

model.compile(
    loss=CRF.sparse_loss,
    optimizer=Adam(learning_rate),
    metrics=[CRF.sparse_accuracy]
)


class NamedEntityRecognizer(ViterbiDecoder):
    """命名实体识别器
    """
    def recognize(self, text):
        tokens = tokenizer.tokenize(text, maxlen=512)
        mapping = tokenizer.rematch(text, tokens)
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        nodes = model.predict([token_ids, segment_ids])[0]
        labels = self.decode(nodes)
        entities, starting = [], False
        for i, label in enumerate(labels):
            if label > 0:
                if label % 2 == 1:
                    starting = True
                    entities.append([[i], categories[(label - 1) // 2]])
                elif starting:
                    entities[-1][0].append(i)
                else:
                    starting = False
            else:
                starting = False
        return [(mapping[w[0]][0], mapping[w[-1]][-1], l) for w, l in entities]


NER = NamedEntityRecognizer(trans=K.eval(CRF.trans), starts=[0], ends=[0])


def evaluate(data):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in tqdm(data, ncols=100):
        R = set(NER.recognize(d[0]))
        T = set([tuple(i) for i in d[1:]])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_f1 = 0

    def on_epoch_end(self, epoch, logs=None):
        trans = K.eval(CRF.trans)
        NER.trans = trans
        print(NER.trans)
        f1, precision, recall = evaluate(valid_data)
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights('./best_model_cluener_crf.weights')
        print(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )


def predict_to_file(in_file, out_file):
    """预测到文件
    可以提交到 https://www.cluebenchmarks.com/ner.html
    """
    fw = open(out_file, 'w', encoding='utf-8')
    with open(in_file) as fr:
        for l in tqdm(fr):
            l = json.loads(l)
            l['label'] = {}
            for start, end, label in NER.recognize(l['text']):
                if label not in l['label']:
                    l['label'][label] = {}
                entity = l['text'][start:end + 1]
                if entity not in l['label'][label]:
                    l['label'][label][entity] = []
                l['label'][label][entity].append([start, end])
            l = json.dumps(l, ensure_ascii=False)
            fw.write(l + '\n')
    fw.close()


if __name__ == '__main__':

    evaluator = Evaluator()
    train_generator = data_generator(train_data, batch_size)

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )

else:

    model.load_weights('./best_model_cluener_crf.weights')
    NER.trans = K.eval(CRF.trans)
    # predict_to_file('/root/ner/cluener/test.json', 'cluener_test.json')
