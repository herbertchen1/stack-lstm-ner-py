# stack-lstm-ner-py
Shift-reduce parser, ner, pytorch

See the paper here: http://arxiv.org/pdf/1603.01360v1.pdf

The default running commands for NER and POS tagging, and NP Chunking are:

- Named Entity Recognition (NER):
```
python tbner.py --train_file ~/data/ner/train.txt --dev_file ~/data/ner/testa.txt --test_file ~/data/ner/testb.txt --checkpoint ./checkpoint/ner_ --caseless --fine_tune --high_way --co_train --emb_file ~/embedding/glove.6B.100d.txt
```

OR

```
nohup python tbner.py --train_file ~/data/ner/train.txt --dev_file ~/data/ner/testa.txt --test_file ~/data/ner/testb.txt --checkpoint ./checkpoint/ner_ --caseless --fine_tune --high_way --co_train --emb_file ~/embedding/glove.6B.100d.txt &
```