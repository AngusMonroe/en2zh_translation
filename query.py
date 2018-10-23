import torch
from torch.autograd import Variable
from torch import optim
from model import Encoder, Decoder, Seq2Seq
import spacy
import jieba
from torchtext.data import Field, BucketIterator, interleave_keys, Dataset
from torchtext.datasets.translation import TranslationDataset

train_sentence_path = 'Data/train_sentences_10000.en-zh'
val_sentence_path = 'Data/validation_sentences_8000.en-zh'
# model = torch.load(f='.save/2018-10-09T05_53_seq2seq_100.pt', map_location=lambda storage, loc: storage)
# print(model)


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

    val_iter, *_ = BucketIterator.splits(
        (val_dataset, ), batch_size=batch_size, repeat=False, sort_key=lambda x: interleave_keys(len(x.src), len(x.trg)))
    print('Training Iterators Built!')
    return val_iter, val_dataset, ZH, EN


def evaluate(model, val_iter):
    model.eval()
    for b, batch in enumerate(val_iter):
        src, len_src = batch.src
        trg, len_trg = batch.trg
        with torch.no_grad():
            src = Variable(src.data.cuda())
            trg = Variable(trg.data.cuda())
        output = model(src, trg, teacher_forcing_ratio=0.0)
        yield output


def query():
    batch_size = 1
    # hidden_size = 512
    # embed_size = 256
    # assert torch.cuda.is_available()

    print("[!] preparing dataset...")
    val_iter, val_dataset, ZH, EN = load_dataset(batch_size=batch_size)
    en_vocab_size, zh_vocab_size = len(EN.vocab), len(ZH.vocab)
    # print("[VALIDATION]:%d (dataset:%d)"
    #       % (len(val_iter), len(val_iter.dataset)))
    print("[EN_vocab]:%d [ZH_vocab]:%d" % (en_vocab_size, zh_vocab_size))

    print("[!] Instantiating models...")
    # encoder = Encoder(en_vocab_size, embed_size, hidden_size,
    #                   n_layers=2, dropout=0.5)
    # decoder = Decoder(embed_size, hidden_size, zh_vocab_size,
    #                   n_layers=1, dropout=0.5)
    # seq2seq = Seq2Seq(encoder, decoder).cpu()
    # seq2seq.load('.save/2018-10-20T05_11_seq2seq_100.ml')
    # # optimizer = optim.Adam(seq2seq.parameters(), lr=args.lr)
    seq2seq = torch.load('.save/2018-10-20T05_11_seq2seq_100.ml')
    print(seq2seq)

    src_sentences = [example.src for example in val_dataset.examples]
    trg_sentences = [example.trg for example in val_dataset.examples]

    test_n = 10
    result_trg = []

    output = evaluate(seq2seq, val_iter)
    for i in range(test_n):
        sample = next(output)
        confidence, prediction = torch.max(sample, dim=2)
        result_trg.append([ZH.vocab.itos[index.item()] for index in prediction])
        # print(' '.join(ZH.vocab.itos[index.item()] for index in prediction))
        # print(src_sentences[i])
        # print(trg_sentences[i], '\n')
    return result_trg, trg_sentences

if __name__ == "__main__":
    result, ground_truth = query()
