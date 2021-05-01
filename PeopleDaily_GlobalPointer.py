#! -*- coding: utf-8 -*-
# 用GlobalPointer做中文命名实体识别
# 数据集 http://s3.bmio.net/kashgari/china-people-daily-ner-corpus.tar.gz

import numpy as np
from bert4keras.backend import keras, K
from bert4keras.backend import multilabel_categorical_crossentropy
from bert4keras.layers import GlobalPointer
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open, to_array
from keras.models import Model
from tqdm import tqdm

maxlen = 256
epochs = 10
batch_size = 16
learning_rate = 2e-5
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
        f = f.read()
        for l in f.split('\n\n'):
            if not l:
                continue
            d = ['']
            for i, c in enumerate(l.split('\n')):
                char, flag = c.split(' ')
                d[0] += char
                if flag[0] == 'B':
                    d.append([i, i, flag[2:]])
                    categories.add(flag[2:])
                elif flag[0] == 'I':
                    d[-1][1] = i
            D.append(d)
    return D


# 标注数据
train_data = load_data('/root/ner/china-people-daily-ner-corpus/example.train')
valid_data = load_data('/root/ner/china-people-daily-ner-corpus/example.dev')
test_data = load_data('/root/ner/china-people-daily-ner-corpus/example.test')
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
            labels = np.zeros((len(categories), maxlen, maxlen))
            for start, end, label in d[1:]:
                if start in start_mapping and end in end_mapping:
                    start = start_mapping[start]
                    end = end_mapping[end]
                    label = categories.index(label)
                    labels[label, start, end] = 1
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels[:, :len(token_ids), :len(token_ids)])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels, seq_dims=3)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


def global_pointer_crossentropy(y_true, y_pred):
    """给GlobalPointer设计的交叉熵
    """
    bh = K.prod(K.shape(y_pred)[:2])
    y_true = K.reshape(y_true, (bh, -1))
    y_pred = K.reshape(y_pred, (bh, -1))
    return K.mean(multilabel_categorical_crossentropy(y_true, y_pred))


def global_pointer_f1_score(y_true, y_pred):
    """给GlobalPointer设计的F1
    """
    y_pred = K.cast(K.greater(y_pred, 0), K.floatx())
    return 2 * K.sum(y_true * y_pred) / K.sum(y_true + y_pred)


model = build_transformer_model(config_path, checkpoint_path)
output = GlobalPointer(len(categories), 64)(model.output)

model = Model(model.input, output)
model.summary()

model.compile(
    loss=global_pointer_crossentropy,
    optimizer=Adam(learning_rate),
    metrics=[global_pointer_f1_score]
)


class NamedEntityRecognizer(object):
    """命名实体识别器
    """
    def recognize(self, text, threshold=0):
        tokens = tokenizer.tokenize(text, maxlen=512)
        mapping = tokenizer.rematch(text, tokens)
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        scores = model.predict([token_ids, segment_ids])[0]
        scores[:, [0, -1]] -= np.inf
        scores[:, :, [0, -1]] -= np.inf
        entities = []
        for l, start, end in zip(*np.where(scores > threshold)):
            entities.append(
                (mapping[start][0], mapping[end][-1], categories[l])
            )
        return entities


NER = NamedEntityRecognizer()


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
        f1, precision, recall = evaluate(valid_data)
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights('./best_model_peopledaily_globalpointer.weights')
        print(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )
        f1, precision, recall = evaluate(test_data)
        print(
            'test:  f1: %.5f, precision: %.5f, recall: %.5f\n' %
            (f1, precision, recall)
        )


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

    model.load_weights('./best_model_peopledaily_globalpointer.weights')
