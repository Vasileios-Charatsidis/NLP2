import sys
import common
import string


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print 'Usage: python find_shortest_sentence.py sentences_file sentence_count'
        sys.exit()

    sentence_count = int(sys.argv[2])
    sentence_file = common.open_utf(sys.argv[1], 'r')
    shortest_len = float('inf')
    shortest_sentence = 0
    sentence_no = 0
    for sentence in sentence_file:
        sentence_len = len(sentence.strip(string.whitespace).split())
        if shortest_len > sentence_len:
            shortest_len = sentence_len
            shortest_sentence = sentence_no
        sentence_no += 1
        if sentence_no >= sentence_count:
            break
    print shortest_sentence
