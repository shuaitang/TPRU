CHANGELOG
---------

v0.3:
 - library is now installable via pip
 - fix occasional problems with UTF-8 whitespace and new lines in learn_bpe and apply_bpe.
   - do not silently convert UTF-8 newline characters into "\n"
   - do not silently convert UTF-8 whitespace characters into " "
   - UTF-8 whitespace and newline characters are now considered part of a word, and segmented by BPE

v0.2:
 - different, more consistent handling of end-of-word token (commit a749a7) (https://github.com/rsennrich/subword-nmt/issues/19)
 - allow passing of vocabulary and frequency threshold to apply_bpe.py, preventing the production of OOV (or rare) subword units (commit a00db)
 - made learn_bpe.py deterministic (commit 4c54e)
 - various changes to make handling of UTF more consistent between Python versions
 - new command line arguments for apply_bpe.py:
   - '--glossaries' to prevent given strings from being affected by BPE
   - '--merges' to apply a subset of learned BPE operations
 - new command line arguments for learn_bpe.py:
   - '--dict-input': rather than raw text file, interpret input as a frequency dictionary (as created by get_vocab.py).


v0.1:
 - consistent cross-version unicode handling
 - all scripts are now deterministic
