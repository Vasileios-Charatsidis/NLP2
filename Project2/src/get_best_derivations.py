import sys
import os
import common


error = 'Usage: python compose_fsts.py sentence_fsts_folder pt_fsts_folder output_folder'
derivations_count = 100

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print error
        sys.exit()

    sentence_fsts_dir = sys.argv[1]
    pt_fsts_dir = sys.argv[2]
    output_dir = sys.argv[3]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sentence_fst_fnames = common.list_filter_sort_filenames(sentence_fsts_dir, lambda f: f.rfind('bin') > 0)
    pt_fst_fnames = common.list_filter_sort_filenames(pt_fsts_dir, lambda f: f.rfind('bin') > 0)

    for sentence_fst_fname, pt_fst_fname in zip(sentence_fst_fnames, pt_fst_fnames):
        sentence_no = int(sentence_fst_fname[sentence_fst_fname.rfind('.')+1:])
        pt_no = int(pt_fst_fname[pt_fst_fname.rfind('.')+1:])
        assert(sentence_no == pt_no)
        output_fst_fname = common.make_path_name(output_dir, 'fst_bin', sentence_no)
        common.compose_fsts(sentence_fst_fname, pt_fst_fname, output_fst_fname)
        best_fst_fname = common.make_path_name(output_dir, 'best_fst_bin', sentence_no)
        common.get_best_fst(output_fst_fname, best_fst_fname, derivations_count)
        best_derivations_fname = common.make_path_name(output_dir, 'best_derivations', sentence_no)
        common.get_best_derivations(best_fst_fname, best_derivations_fname)
        # h stands for human readable
        best_derivations_h_fname = common.make_path_name(output_dir, 'monotone.{0:d}best'.format(derivations_count), sentence_no)
        common.get_best_derivations_h(best_derivations_fname, best_derivations_h_fname)
