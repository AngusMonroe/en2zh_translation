import re
import spacy
from torchtext.data import Field, BucketIterator, interleave_keys
from torchtext.datasets import Multi30k


def load_dataset(batch_size):
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')
    url = re.compile('(<url>.*</url>)')

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(url.sub('@URL@', text))]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(url.sub('@URL@', text))]

    DE = Field(tokenize=tokenize_de, include_lengths=True,
               init_token='<sos>', eos_token='<eos>')
    EN = Field(tokenize=tokenize_en, include_lengths=True,
               init_token='<sos>', eos_token='<eos>')
    train, val, test = Multi30k.splits(exts=('.de', '.en'), fields=(DE, EN))
    # mt_train = datasets.TranslationDataset(
    # path = 'data/mt/wmt16-ende.train', exts = ('.en', '.de'),
    # fields = (src, trg))
    DE.build_vocab(train.src, min_freq=2)
    EN.build_vocab(train.trg, max_size=10000)
    # train_iter = BucketIterator(dataset=train, batch_size=batch_size, sort_key=lambda x: interleave_keys(len(x.src), len(x.trg)))
    train_iter, val_iter, test_iter = BucketIterator.splits(
            (train, val, test), batch_size=batch_size, repeat=False, sort_key=lambda x: interleave_keys(len(x.src), len(x.trg)))
    return train_iter, val_iter, test_iter, DE, EN
