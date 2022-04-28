import os
import sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + './../..')
import argparse
from argparse import ArgumentParser


ARG_PARSER = ArgumentParser()

ARG_PARSER.add_argument('--batch_size', default=100, type=int)
ARG_PARSER.add_argument('--test_size', default=0.2, type=int)
ARG_PARSER.add_argument('--val_size', default=0.1, type=int)

ARG_PARSER.add_argument('--num_epochs', default=50, type=int)
ARG_PARSER.add_argument('--patience', default=5, type=int)


# ARG_PARSER.add_argument('--cond_seq_len', default=39, type=int)#473
# ARG_PARSER.add_argument('--proc_seq_len', default=24, type=int)#490
# ARG_PARSER.add_argument('--med_seq_len', default=280, type=int)#565
# ARG_PARSER.add_argument('--out_seq_len', default=146, type=int)
# ARG_PARSER.add_argument('--chart_seq_len', default=118, type=int)#54

# ARG_PARSER.add_argument('--cond_vocab_size', default=1424, type=int)#607#625
# ARG_PARSER.add_argument('--proc_vocab_size', default=152, type=int)#355#347
# ARG_PARSER.add_argument('--med_vocab_size', default=274, type=int)#340#378
# ARG_PARSER.add_argument('--out_vocab_size', default=72, type=int)
# ARG_PARSER.add_argument('--chart_vocab_size', default=76, type=int)#74,75

ARG_PARSER.add_argument('--embedding_size', default=52, type=float)
ARG_PARSER.add_argument('--rnn_size', default=512, type=float)
ARG_PARSER.add_argument('--lrn_rate', default=0.0001, type=float)


args = ARG_PARSER.parse_args(args=[])