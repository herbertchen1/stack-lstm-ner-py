# stack-lstm-ner-py
Shift-reduce parser, ner, python

See the paper here: http://arxiv.org/pdf/1603.01360v1.pdf


The default running commands for NER and POS tagging, and NP Chunking are:

- Named Entity Recognition (NER):
```
python train_wc.py --train_file ./data/ner/train.txt --dev_file ./data/ner/testa.txt --test_file ./data/ner/testb.txt --checkpoint ./checkpoint/ner_ --caseless --fine_tune --high_way --co_train
```

- Part-of-Speech (POS) Tagging:
```
python train_wc.py --train_file ./data/pos/train.txt --dev_file ./data/pos/testa.txt --test_file ./data/pos/testb.txt --eva_matrix a --checkpoint ./checkpoint/pos_ --lr 0.015 --caseless --fine_tune --high_way --co_train
```

- Noun Phrase (NP) Chunking:
```
python train_wc.py --train_file ./data/np/train.txt.iobes --dev_file ./data/np/testa.txt.iobes --test_file ./data/np/testb.txt.iobes --checkpoint ./checkpoint/np_ --caseless --fine_tune --high_way --co_train --least_iters 100
```

For other datasets or tasks, you may wanna try different stopping parameters, especially, for smaller dataset, you may want to set ```least_iters``` to a larger value; and for some tasks, if the speed of loss decreasing is too slow, you may want to increase ```lr```.