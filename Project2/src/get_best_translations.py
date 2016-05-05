import sys
import os
import common


error = 'Usage: python get_best_translations.py derivations_dir output_dir'


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print error
        sys.exit()

    derivations_dir = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    derivations_fnames = common.list_filter_sort_filenames(derivations_dir, lambda f: f.rfind('derivations.') > 0)

    best_translations = common.open_utf(os.path.join(output_dir, 'monotone.trans'), 'w')
    best_derivations = common.open_utf(os.path.join(output_dir, 'monotone.der'), 'w')
    for derivations_fname in derivations_fnames:
        best_tr, best_der = common.get_best_translation_with_best_derivation(derivations_fname)
        best_translations.write(best_tr + '\n')
        best_derivations.write(best_der + '\n')
    best_translations.close()
    best_derivations.close()
