# -*- coding: utf-8 -*-
# @Time    : 2020/3/16 7:39 下午
# @Author  : Xiachong Feng
# @File    : my_field.py
# @Software: PyCharm

from collections import Counter, OrderedDict
from itertools import chain
import six
import torch

from torchtext.data import RawField
from torchtext.data.dataset import Dataset
from torchtext.data.pipeline import Pipeline
from torchtext.data.utils import get_tokenizer, dtype_to_attr, is_tokenizer_serializable
from torchtext.vocab import Vocab

from more_itertools import locate


class MySrcField(RawField):
    """Defines a datatype together with instructions for converting to Tensor.

    Field class models common text processing datatypes that can be represented
    by tensors.  It holds a Vocab object that defines the set of possible values
    for elements of the field and their corresponding numerical representations.
    The Field object also holds other parameters relating to how a datatype
    should be numericalized, such as a tokenization method and the kind of
    Tensor that should be produced.

    If a Field is shared between two columns in a dataset (e.g., question and
    answer in a QA dataset), then they will have a shared vocabulary.

    Attributes:
        sequential: Whether the datatype represents sequential data. If False,
            no tokenization is applied. Default: True.
        use_vocab: Whether to use a Vocab object. If False, the data in this
            field should already be numerical. Default: True.
        init_token: A token that will be prepended to every example using this
            field, or None for no initial token. Default: None.
        eos_token: A token that will be appended to every example using this
            field, or None for no end-of-sentence token. Default: None.
        fix_length: A fixed length that all examples using this field will be
            padded to, or None for flexible sequence lengths. Default: None.
        dtype: The torch.dtype class that represents a batch of examples
            of this kind of data. Default: torch.long.
        preprocessing: The Pipeline that will be applied to examples
            using this field after tokenizing but before numericalizing. Many
            Datasets replace this attribute with a custom preprocessor.
            Default: None.
        postprocessing: A Pipeline that will be applied to examples using
            this field after numericalizing but before the numbers are turned
            into a Tensor. The pipeline function takes the batch as a list, and
            the field's Vocab.
            Default: None.
        lower: Whether to lowercase the text in this field. Default: False.
        tokenize: The function used to tokenize strings using this field into
            sequential examples. If "spacy", the SpaCy tokenizer is
            used. If a non-serializable function is passed as an argument,
            the field will not be able to be serialized. Default: string.split.
        tokenizer_language: The language of the tokenizer to be constructed.
            Various languages currently supported only in SpaCy.
        include_lengths: Whether to return a tuple of a padded minibatch and
            a list containing the lengths of each examples, or just a padded
            minibatch. Default: False.
        batch_first: Whether to produce tensors with the batch dimension first.
            Default: False.
        pad_token: The string token used as padding. Default: "<pad>".
        unk_token: The string token used to represent OOV words. Default: "<unk>".
        pad_first: Do the padding of the sequence at the beginning. Default: False.
        truncate_first: Do the truncating of the sequence at the beginning. Default: False
        stop_words: Tokens to discard during the preprocessing step. Default: None
        is_target: Whether this field is a target variable.
            Affects iteration over batches. Default: False
    """

    vocab_cls = Vocab
    # Dictionary mapping PyTorch tensor dtypes to the appropriate Python
    # numeric type.
    dtypes = {
        torch.float32: float,
        torch.float: float,
        torch.float64: float,
        torch.double: float,
        torch.float16: float,
        torch.half: float,

        torch.uint8: int,
        torch.int8: int,
        torch.int16: int,
        torch.short: int,
        torch.int32: int,
        torch.int: int,
        torch.int64: int,
        torch.long: int,
    }

    ignore = ['dtype', 'tokenize']

    def __init__(self, sequential=True, use_vocab=True, init_token=None,
                 eos_token=None, fix_length=None, dtype=torch.long,
                 preprocessing=None, postprocessing=None, lower=False,
                 tokenize=None, tokenizer_language='en', include_lengths=False,
                 batch_first=False, pad_token="<pad>", unk_token="<unk>", sent_split="[SEP]",
                 pad_first=False, truncate_first=False, stop_words=None,
                 is_target=False):
        self.sequential = sequential
        self.use_vocab = use_vocab
        self.init_token = init_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.fix_length = fix_length
        self.dtype = dtype
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing
        self.lower = lower
        # store params to construct tokenizer for serialization
        # in case the tokenizer isn't picklable (e.g. spacy)
        self.tokenizer_args = (tokenize, tokenizer_language)
        self.tokenize = get_tokenizer(tokenize, tokenizer_language)
        self.include_lengths = include_lengths
        self.batch_first = batch_first
        self.pad_token = pad_token if self.sequential else None
        self.pad_first = pad_first
        self.truncate_first = truncate_first
        self.sent_split = sent_split
        try:
            self.stop_words = set(stop_words) if stop_words is not None else None
        except TypeError:
            raise ValueError("Stop words must be convertible to a set")
        self.is_target = is_target

    def __getstate__(self):
        str_type = dtype_to_attr(self.dtype)
        if is_tokenizer_serializable(*self.tokenizer_args):
            tokenize = self.tokenize
        else:
            # signal to restore in `__setstate__`
            tokenize = None
        attrs = {k: v for k, v in self.__dict__.items() if k not in self.ignore}
        attrs['dtype'] = str_type
        attrs['tokenize'] = tokenize

        return attrs

    def __setstate__(self, state):
        state['dtype'] = getattr(torch, state['dtype'])
        if not state['tokenize']:
            state['tokenize'] = get_tokenizer(*state['tokenizer_args'])
        self.__dict__.update(state)

    def __hash__(self):
        # we don't expect this to be called often
        return 42

    def __eq__(self, other):
        if not isinstance(other, RawField):
            return False

        return self.__dict__ == other.__dict__

    def preprocess(self, x):
        """Load a single example using this field, tokenizing if necessary.

        If the input is a Python 2 `str`, it will be converted to Unicode
        first. If `sequential=True`, it will be tokenized. Then the input
        will be optionally lowercased and passed to the user-provided
        `preprocessing` Pipeline."""
        if (six.PY2 and isinstance(x, six.string_types)
                and not isinstance(x, six.text_type)):
            x = Pipeline(lambda s: six.text_type(s, encoding='utf-8'))(x)
        if self.sequential and isinstance(x, six.text_type):
            x = self.tokenize(x.rstrip('\n'))
        if self.lower:
            x = Pipeline(six.text_type.lower)(x)
        if self.sequential and self.use_vocab and self.stop_words is not None:
            x = [w for w in x if w not in self.stop_words]
        if self.preprocessing is not None:
            return self.preprocessing(x)
        else:
            return x

    def process(self, batch, device=None):
        """ Process a list of examples to create a torch.Tensor."""

        node_batch_data_padded, node_batch_data_len, batch_node_nums, max_node_num, max_node_len = self.pad(batch)
        node_batch_data, node_batch_data_len, recover_idx = self.numericalize(node_batch_data_padded,
                                                                              node_batch_data_len, device=device)
        return {"data": node_batch_data, "len": node_batch_data_len, "recover": recover_idx,
                "batch_node_nums": batch_node_nums, "max_node_num": max_node_num, "max_node_len": max_node_len}

    def pad(self, minibatch):
        """Pad a batch of examples using this field."""

        minibatch = list(minibatch)
        batch_hier_data = list()
        batch_node_nums = list()
        batch_node_lens = list()
        for example in minibatch:
            res = list()
            index_sent_split_list = list(locate(example, lambda a: a == self.sent_split))
            index_sent_split_list = [-1] + index_sent_split_list
            for index in range(len(index_sent_split_list) - 1):
                res.append(example[index_sent_split_list[index] + 1:index_sent_split_list[index + 1] + 1])
            batch_hier_data.append(res)
            batch_node_nums.append(len(res))
            batch_node_lens.append(list(map(len, res)))
        max_node_num = max(batch_node_nums)
        max_node_len = max(map(max, batch_node_lens))

        # pad
        batch_hier_data_padded = list()
        for hier_data in batch_hier_data:
            hier_data = hier_data + [[self.pad_token]] * (max_node_num - len(hier_data))
            hier_data_padded = [node + [self.pad_token] * (max_node_len - len(node)) for node in hier_data]
            batch_hier_data_padded.append(hier_data_padded)
        batch_node_lens_padded = [word_nums + [0] * (max_node_num - len(word_nums)) for word_nums in batch_node_lens]

        # reduce dim (node-level)
        node_batch_data_padded = [node for hier_data_padded in batch_hier_data_padded for node in hier_data_padded]
        node_batch_data_len = [length for node_lens_padded in batch_node_lens_padded for length in node_lens_padded]

        return node_batch_data_padded, node_batch_data_len, batch_node_nums, max_node_num, max_node_len

    def numericalize(self, node_batch_data_padded, node_batch_data_len, device=None):
        """Turn a batch of examples that use this field into a Variable."""

        # convert to ids
        node_batch_data_ids = [[self.vocab.stoi[x] for x in one] for one in node_batch_data_padded]

        # convert to tensor
        node_batch_data_len = torch.tensor(node_batch_data_len, dtype=self.dtype, device=device)
        node_batch_data = torch.tensor(node_batch_data_ids, dtype=self.dtype, device=device)

        # sort
        node_batch_data_len, perm_idx = node_batch_data_len.sort(0, descending=True)
        node_batch_data = node_batch_data[perm_idx]
        _, recover_idx = perm_idx.sort(0, descending=False)
        # a = node_batch_data[recover_idx]

        if self.sequential and not self.batch_first:
            node_batch_data.t_()
        if self.sequential:
            node_batch_data = node_batch_data.contiguous()

        return node_batch_data, node_batch_data_len, recover_idx

    def build_vocab(self, *args, **kwargs):
        """Construct the Vocab object for this field from one or more datasets.

        Arguments:
            Positional arguments: Dataset objects or other iterable data
                sources from which to construct the Vocab object that
                represents the set of possible values for this field. If
                a Dataset object is provided, all columns corresponding
                to this field are used; individual columns can also be
                provided directly.
            Remaining keyword arguments: Passed to the constructor of Vocab.
        """
        counter = Counter()
        sources = []
        for arg in args:
            if isinstance(arg, Dataset):
                sources += [getattr(arg, name) for name, field in
                            arg.fields.items() if field is self]
            else:
                sources.append(arg)
        for data in sources:
            for x in data:
                if not self.sequential:
                    x = [x]
                try:
                    counter.update(x)
                except TypeError:
                    counter.update(chain.from_iterable(x))
        specials = list(OrderedDict.fromkeys(
            tok for tok in [self.unk_token, self.pad_token, self.init_token,
                            self.eos_token] + kwargs.pop('specials', [])
            if tok is not None))
        self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)
