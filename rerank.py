import argparse
import codecs
from collections import Counter
from itertools import islice


def rerank(gen_summaries):
    res = []

    def process_one(sequence):
        gram_cnt = Counter(_make_n_gram(sequence))
        return gram_cnt

    for index, gen_summary in enumerate(gen_summaries):
        gram_cnt = process_one(gen_summary)
        res.append(sum(count - 1 for gram, count in gram_cnt.items() if count > 1))
    min_ngram_count = min(res)
    for index, ngram_count in enumerate(res):
        if ngram_count == min_ngram_count:
            best = index
            break
    best_summary = gen_summaries[best]
    return best_summary


def _make_n_gram(sequence, n=2):
    return (tuple(sequence[i:i + n]) for i in range(len(sequence) - (n - 1)))


if __name__ == "__main__":
    # for one test sample ,we have gen_summaries
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=str, default="",
                        help='candidate file')
    parser.add_argument('-n', type=int, default=4,
                        help="top n")
    args = parser.parse_args()

    out = codecs.open(args.c.replace(".txt", "") + "_rerank.txt", "w", "utf-8")

    with codecs.open(args.c, "r", "utf-8") as f:
        lines = f.readlines()

        for index in range(0, 819):
            start = index * args.n
            end = index * args.n + args.n
            _lines = lines[start:end]
            gen_summaries = [line.strip().split() for line in _lines]
            best_summary = rerank(gen_summaries)
            out.write(" ".join(best_summary) + "\n")
