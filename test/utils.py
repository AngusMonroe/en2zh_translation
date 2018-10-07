import re
import spacy
from bs4 import BeautifulSoup
from spacy.lang.zh import Chinese
from torchtext.data import Field, BucketIterator, interleave_keys, Dataset
from torchtext.datasets import IWSLT

# train_file  = '../Data/ai_challenger_MTEnglishtoChinese_trainingset_20180827.txt'
train_file = '../Data/first_10000_training_samples.txt'
val_en_file = '../Data/ai_challenger_MTEnglishtoChinese_validationset_20180823_en.sgm'
val_zh_file = '../Data/ai_challenger_MTEnglishtoChinese_validationset_20180823_zh.sgm'
test_en_file = '../Data/ai_challenger_MTEnglishtoChinese_testA_20180827_en.sgm'


def extract_data_from_sgm(sgm_filename, cols):
    text = open(sgm_filename, 'rt', encoding='utf-8').read()
    bs = BeautifulSoup(text, 'xml')
    lines = bs.find_all('seg')
    samples = [line.get_text().split('\t') for line in lines]
    result = list(zip(*samples))
    assert len(result) == cols, ('Wrong cols argument, found %d while require %d' % (
        len(result), cols))
    return result


class sentence_translation(object):
    def __init__(self, docID, senID, en, zh):
        super(sentence_translation, self).__init__()
        self.docID = docID
        self.senID = senID
        self.src = en
        self.trg = zh


def load_dataset(batch_size, debug=True):
    spacy_en = spacy.load('en')
    spacy_zh = Chinese()

    def tokenize_en(line):
        return [token.text for token in spacy_zh.tokenizer(line)]

    def tokenize_zh(line):
        return [token.text for token in spacy_en.tokenizer(line)]

    EN = Field(tokenize=tokenize_en, include_lengths=True,
               init_token='<sos>', eos_token='<eos>')
    ZH = Field(tokenize=tokenize_zh, include_lengths=True,
               init_token='<sos>', eos_token='<eos>')

    lines = open(train_file, 'rt', encoding='utf-8').read().splitlines()
    train_samples = [line.split('\t') for line in lines]
    train_docID, train_senID, train_en, train_zh = zip(*train_samples)

    val_docID, val_senID, val_en = extract_data_from_sgm(val_en_file, cols=3)
    val_zh, = extract_data_from_sgm(val_zh_file, cols=1)

    test_docID, test_senID, test_en = extract_data_from_sgm(
        test_en_file, cols=3)

    if debug:
        debug_info_size = 10
        print('\n[Debug] First %d training examples:\n' % debug_info_size)
        for i in range(debug_info_size):
            print(train_docID[i], train_senID[i], train_en[i], train_zh[i])
        print('\n[Debug] First %d validation examples:\n' % debug_info_size)
        for i in range(debug_info_size):
            print(val_docID[i], val_senID[i], val_en[i], val_zh[i])
        print('\n[Debug] First %d test examples:\n' % debug_info_size)
        for i in range(debug_info_size):
            print(test_en[i])

    train_examples = [
        sentence_translation(
            train_docID[i], train_senID[i], train_en[i], train_zh[i])
        for i in range(len(train_docID))]
    val_examples = [
        sentence_translation(
            val_docID[i], val_senID[i], val_en[i], val_zh[i])
        for i in range(len(val_docID))]

    print("Train size = %d" % len(train_examples))
    print("Eval size = %d" % len(val_examples))

    train_dataset = Dataset(train_examples, {'src': EN, 'trg': ZH})
    val_dataset = Dataset(val_examples, {'src': EN, 'trg': ZH})
    print('Datasets Built!')

    EN.build_vocab(train_dataset.src, min_freq=2)
    ZH.build_vocab(train_dataset.trg, max_size=10000)
    print('Vocabularies Built!')

    train_iter, val_iter = BucketIterator.splits(
        (train_dataset, val_dataset), batch_size=batch_size, repeat=False, sort_key=lambda x: interleave_keys(len(x.src), len(x.trg)))
    print('Training Iterators Built!')
    return train_iter, val_iter, ZH, EN


if __name__ == '__main__':
    test_batch_size = 64
    train_iter, val_iter, ZH, EN = load_dataset(test_batch_size, True)
    batches = list(train_iter)
    first_batch = batches[0]
    src, len_src = first_batch.src
    trg, len_trg = first_batch.trg
    sentence = src[12]
    print(sentence)
    print(' '.join(EN.vocab.itos[index.item()] for index in sentence))
