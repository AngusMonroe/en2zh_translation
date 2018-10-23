import re
import spacy
import dill
from bs4 import BeautifulSoup
from os import path, makedirs
import jieba
from torchtext.data import Field, BucketIterator, interleave_keys, Dataset
from torchtext.datasets.translation import TranslationDataset

data_path = './Data/'
field_path = 'fields/'
train_sentence_path = path.join(data_path, 'train_sentences_%d.en-zh')
val_sentence_path = path.join(data_path, 'validation_sentences_%d.en-zh')


def load_dataset(batch_size, train_maxsize, val_maxsize, debug=True):
    spacy_en = spacy.load('en')

    def tokenize_en(line):
        return [token.text for token in spacy_en.tokenizer(line)]

    def tokenize_zh(line):
        return [token for token in jieba.cut(line)]

    EN = Field(tokenize=tokenize_en, include_lengths=True,
               init_token='<sos>', eos_token='<eos>')
    ZH = Field(tokenize=tokenize_zh, include_lengths=True,
               init_token='<sos>', eos_token='<eos>')

    exts = ['.en', '.zh']
    fields = [
        ('src', EN),
        ('trg', ZH)
    ]
    train_dataset = TranslationDataset(
        train_sentence_path % train_maxsize, exts=exts, fields=fields)
    val_dataset = TranslationDataset(
        val_sentence_path % val_maxsize, exts=exts, fields=fields)
    print('Datasets Built!')

    EN.build_vocab(train_dataset.src, min_freq=2)
    ZH.build_vocab(train_dataset.trg, max_size=100000)
    print('Vocabularies Built!')

    # en_field_path = path.join(data_path, field_path,  'train_%d_val_%d_field_en') % (
    #     train_maxsize, val_maxsize)
    # zh_field_path = path.join(data_path, field_path, 'train_%d_val_%d_field_zh') % (
    #     train_maxsize, val_maxsize)
    # try:
    #     if not path.exists(en_field_path):
    #         makedirs(path.join(data_path, field_path))
    #     dill.dump(EN, open(en_field_path, 'wb'))
    #     dill.dump(ZH, open(zh_field_path, 'wb'))
    # except OSError as e:
    #     print('OS Error, while storing en and zh Vocabularies')
    #     print(e)

    # first parameter should be a iterable object, and the return result will have same len with it
    train_iter, val_iter = BucketIterator.splits(
        (train_dataset, val_dataset), batch_size=batch_size, repeat=False, sort_key=lambda x: interleave_keys(len(x.src), len(x.trg)))
    print('Training Iterators Built!')
    return train_iter, val_iter, ZH, EN


if __name__ == '__main__':
    test_batch_size = 64
    train_iter, *_, EN = load_dataset(test_batch_size, 1000, 1000, True)
    batch = list(train_iter)[0]
    src, len_src = batch.src
    trg, len_trg = batch.trg
    sentence = src[:, 12]
    print(sentence)
    print(' '.join(EN.vocab.itos[index.item()] for index in sentence))
