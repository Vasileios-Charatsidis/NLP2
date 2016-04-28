import sys
import os
import subprocess


error = 'Usage: python compose_fsts.py sentence_fsts_folder pt_fsts_folder output_folder'


def construct_callee(sentence_fst, pt_fst, output_fst):
    # fstcompile --isymbols=isyms.txt --osymbols=osyms.txt --keep_isymbols --keep_osymbols text.fst binary.fst
    callee = ['fstcompose']
    callee.append(sentence_fst)
    callee.append(pt_fst)
    callee.append(output_fst)
    return callee


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print error
        sys.exit()

    sentence_fsts_dir = sys.argv[1]
    pt_fsts_dir = sys.argv[2]
    output_dir = sys.argv[3]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get all files from the sentence dir
    sentence_fst_fnames = [f for f in os.listdir(sentence_fsts_dir) if f.find('fst.fst') >= 0]
    # Sort them based on corresponding sentence number
    sentence_fst_fnames.sort(key=lambda f: int(f[:f.find('fst.fst')]))

    # Get all files from the sentence dir
    pt_fst_fnames = [f for f in os.listdir(pt_fsts_dir) if f.find('fst.fst') >= 0]
    # Sort them based on corresponding sentence number
    pt_fst_fnames.sort(key=lambda f: int(f[:f.find('f')]))

    for sentence_fst_fname, pt_fst_fname in zip(sentence_fst_fnames, pt_fst_fnames):
        sentence_no = sentence_fst_fname[:sentence_fst_fname.find('fst.fst')]
        sentence_fst_path = os.path.join(sentence_fsts_dir, sentence_fst_fname)
        pt_fst_path = os.path.join(pt_fsts_dir, pt_fst_fname)
        output_fst_path = os.path.join(output_dir, sentence_no + 'fst.fst')
        callee = construct_callee(sentence_fst_path, pt_fst_path, output_fst_path)
        subprocess.call(callee)