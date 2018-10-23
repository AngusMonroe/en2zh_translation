import os
import math
import argparse
import torch
from torch import optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from torch.nn import functional as F
from model import Encoder, Decoder, Seq2Seq
from utils import load_dataset
import datetime


def get_time_str():
    return datetime.datetime.now().strftime('%Y-%m-%dT%H_%M')


def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-epochs', type=int, default=100,
                   help='number of epochs for train')
    p.add_argument('-batch_size', type=int, default=32,
                   help='number of epochs for train')
    p.add_argument('-lr', type=float, default=0.0001,
                   help='initial learning rate')
    p.add_argument('-grad_clip', type=float, default=10.0,
                   help='initial learning rate')
    p.add_argument('-train_maxsize', type=int, default=100000,
                   help='max training set size')
    p.add_argument('-val_maxsize', type=int, default=8000,
                   help='max validation set size, default is 8000 (using the whole)')
    return p.parse_args()


def evaluate(model, val_iter, ZH_vocab_size, EN, ZH):
    model.eval()
    pad = ZH.vocab.stoi['<pad>']
    total_loss = 0
    for b, batch in enumerate(val_iter):
        src, len_src = batch.src
        trg, len_trg = batch.trg
        with torch.no_grad():
            src = Variable(src.data.cuda())
            trg = Variable(trg.data.cuda())
        output = model(src, trg, teacher_forcing_ratio=0.0)
        loss = F.nll_loss(output[1:].view(-1, ZH_vocab_size),
                          trg[1:].contiguous().view(-1),
                          ignore_index=pad)
        total_loss += loss.data.item()
    return total_loss / len(val_iter)


def train(epoch, model, optimizer, train_iter, ZH_vocab_size, grad_clip, EN, ZH, show_detail=False):
    model.train()
    total_loss = 0
    pad = ZH.vocab.stoi['<pad>']
    for b, batch in enumerate(train_iter):
        src, len_src = batch.src
        trg, len_trg = batch.trg

        if show_detail:
            for i in range(src.size(1)):
                print('len=%d\t' % len_src[i], ' '.join(token for token in [
                      EN.vocab.itos[index.item()] for index in src[:, i]] if token != '<pad>'))
            for i in range(trg.size(1)):
                print('len=%d\t' % len_trg[i], ' '.join(token for token in [
                      ZH.vocab.itos[index.item()] for index in trg[:, i]] if token != '<pad>'))

        src, trg = src.cuda(), trg.cuda()   # trg: (max_seq_len, batch_size)
        optimizer.zero_grad()
        output = model(src, trg)    # (max_seq_len, batch_size, ZH_vocab_size)
        loss = F.nll_loss(output[1:].view(-1, ZH_vocab_size),   # remove '<sos>' using [1:]
                          trg[1:].contiguous().view(-1),
                          ignore_index=pad)
        loss.backward()
        clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.data.item()

        if b % 100 == 0 and b != 0:
            total_loss = total_loss / 100
            print("[%d][loss:%5.2f][pp:%5.2f]" %
                  (b, total_loss, math.exp(total_loss)))
            total_loss = 0


def main(debug=True, show_detail=False):
    args = parse_arguments()
    hidden_size = 512
    embed_size = 256
    assert torch.cuda.is_available()

    print("[!] preparing dataset...")
    train_iter, val_iter, ZH, EN = load_dataset(
        args.batch_size, args.train_maxsize, args.val_maxsize)

    en_vocab_size, zh_vocab_size = len(EN.vocab), len(ZH.vocab)
    print("[TRAIN]:%d (dataset:%d)\t[VALIDATION]:%d (dataset:%d)"
          % (len(train_iter), len(train_iter.dataset),
             len(val_iter), len(val_iter.dataset)))
    print("[EN_vocab]:%d [ZH_vocab]:%d" % (en_vocab_size, zh_vocab_size))

    print("[!] Instantiating models...")
    encoder = Encoder(en_vocab_size, embed_size, hidden_size,
                      n_layers=2, dropout=0.5)
    decoder = Decoder(embed_size, hidden_size, zh_vocab_size,
                      n_layers=1, dropout=0.5)
    seq2seq = Seq2Seq(encoder, decoder).cuda()
    optimizer = optim.Adam(seq2seq.parameters(), lr=args.lr)
    print(seq2seq)

    time_str = get_time_str()
    for epoch in range(1, args.epochs+1):
        train(epoch, seq2seq, optimizer, train_iter,
              zh_vocab_size, args.grad_clip, EN, ZH, show_detail=show_detail)
        val_loss = evaluate(seq2seq, val_iter, zh_vocab_size, EN, ZH)
        print("[Epoch:%d] val_loss:%5.3f | val_pp:%5.2fS"
              % (epoch, val_loss, math.exp(val_loss)))

        if epoch % 10 == 0:
            print("[!] saving model...")
            if not os.path.isdir(".save"):
                os.makedirs(".save")
            save_path = './.save/%s_seq2seq_%d.pt' % (time_str, epoch)
            torch.save(seq2seq, save_path)
    # test_loss = evaluate(seq2seq, test_iter, zh_vocab_size, EN, ZH)
    # print("[TEST] loss:%5.2f" % test_loss)


if __name__ == "__main__":
    try:
        main(debug=True, show_detail=False)
    except KeyboardInterrupt as epoch:
        print("[STOP]", epoch)
