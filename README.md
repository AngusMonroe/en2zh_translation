# en2zh_translation

A Seq2Seq model with attention for neural machine translation in PyTorch.

This implementation focuses on the following features:

- Modular structure to be used in other projects
- Minimal code for readability
- Full utilization of batches and GPU.

This implementation also use BLEU, NIST score and TER to evaluate the effect of the model.

## Model description

* Encoder: Bidirectional GRU
* Decoder: GRU with Attention Mechanism
* Attention: [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)

![](http://www.wildml.com/wp-content/uploads/2015/12/Screen-Shot-2015-12-30-at-1.16.08-PM.png)

## Metric

- BLUE
- NIST
- TER
- METEOR

```
perl mteval-v13a.pl -r example/ref.xml -s example/src.xml -t example/tst.xml
```

Use `xml_transform.py` to transorm sentence txt to xml file.

reference: [https://blog.csdn.net/angus_monroe/article/details/82943162](https://blog.csdn.net/angus_monroe/article/details/82943162)

## Requirements

* GPU & CUDA
* Python3
* PyTorch
* torchtext
* Spacy
* numpy
* Visdom (optional)

## References

Based on the following implementations

* [PyTorch Tutorial](http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
* [@spro/practical-pytorch](https://github.com/spro/practical-pytorch)
* [@AuCson/PyTorch-Batch-Attention-Seq2seq](https://github.com/AuCson/PyTorch-Batch-Attention-Seq2seq)
