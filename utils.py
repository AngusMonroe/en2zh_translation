import re
import spacy
from bs4 import BeautifulSoup
import jieba
from torchtext.data import Field, BucketIterator, interleave_keys, Dataset
from torchtext.datasets.translation import TranslationDataset

train_sentence_path = '../Data/train_sentences_10000.en-zh'
val_sentence_path = '../Data/validation_sentences_8000.en-zh'

# train_file  = '../Data/ai_challenger_MTEnglishtoChinese_trainingset_20180827.txt'
train_file = '../Data/first_10000_training_samples.txt'
val_en_file = '../Data/ai_challenger_MTEnglishtoChinese_validationset_20180823_en.sgm'
val_zh_file = '../Data/ai_challenger_MTEnglishtoChinese_validationset_20180823_zh.sgm'
test_en_file = '../Data/ai_challenger_MTEnglishtoChinese_testA_20180827_en.sgm'


def load_dataset(batch_size, debug=True):
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
        train_sentence_path, exts=exts, fields=fields)
    val_dataset = TranslationDataset(
        val_sentence_path, exts=exts, fields=fields)
    print('Datasets Built!')

    EN.build_vocab(train_dataset.src, min_freq=2)
    ZH.build_vocab(train_dataset.trg, max_size=100000)
    print('Vocabularies Built!')

    train_iter, val_iter = BucketIterator.splits(
        (train_dataset, val_dataset), batch_size=batch_size, repeat=False, sort_key=lambda x: interleave_keys(len(x.src), len(x.trg)))
    print('Training Iterators Built!')
    return train_iter, val_iter, ZH, EN


if __name__ == '__main__':
    test_batch_size = 64
    train_iter, *_ = load_dataset(test_batch_size, True)
    batch = list(train_iter)[0]
    src, len_src = batch.src
    trg, len_trg = batch.trg
    sentence = src[:, 12]
    print(sentence)
    print(' '.join(EN.vocab.itos[index.item()] for index in sentence))
