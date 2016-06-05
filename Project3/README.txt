translation.py and datareader.py provide classes to respectively abstract a translation into a python object, and read n-best lists of translations, references, and pos-tags into translation obejcts

================================================================================

To POS-tag translations run:
python pretag.py sentences.gz out_dir language from_sentence to_sentence process_no

sentences.gz is the file with n-best translations, for example nlp2-dev.1000best.gz
out_dir is a directory name, where the the POS tags for the translations will be saved
language could be either english or german
from_sentence is the first source sentence for which translations should be POS tagged (inclusive)
to_sentence is the last source sentence for which translations should be POS tagged (exclusive)
process_no is the number of processes to be run in parallel. Not sanity check is made in the code. Use responsibly.

Example:
python pretag.py nlp2-dev.1000best.gz tagged_dev german 0 500 4

================================================================================

To concatenate the files with POS tags resulting from pretag.py into a single file use:
concat.sh input_dir output_file

Example:
./concat.sh tagged_dev tagged_dev_file

================================================================================

To tune a MT model use:
python pro.py nbest_translations.gz reference_translations.gz pos_tags src_sentence_prob pro_sample_size basic out_weights

nbest_translations.gz is the file with n-best translations, for example nlp2-dev.1000best.gz
reference_translations.gz is the file with the reference translations, for example nlp2-dev.de.gz
pos_tags is the file with POS tags for the n-best translations
src_sentence_prob is a probability with which the translations for a source sentence to be used when generating training instances for the classifier. To use all sentences set to 1
pro_sample_size is the sample size of the PRO algorithm, e.g. 50
basic can be either True or False. If True, only the baseline model features will be used for tuning, else extra features are added.
out_weights is the name of the file, in which model weights will be written

Example:
python pro.py nlp2-dev.1000best.gz nlp2-dev.de.gz tagged_dev_file 1 50 True basic.weights

================================================================================

To get the best translations of a model use:
python get_best_translations.py translations_file pos_tags_file weights_file out_best_file

translations_file is the file with n-best translations, for example nlp2-dev.1000best.gz
pos_tags_file is the file with POS tags for the n-best translations
weights_file is the file with the weights for the log-linear model to be used
out_best_file is the file in which the best translations will be written

Example:
python get_best_translations.py nlp2-dev.1000best.gz tagged_dev_file basic.weights basic.dev

