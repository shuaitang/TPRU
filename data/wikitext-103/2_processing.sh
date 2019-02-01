for split in train valid test
do 
  ftfy raw/wikitext-103/wiki.$split.tokens \
    -n NFKC \
    -o raw/wikitext-103/wiki.$split.clean.tokens && \
  echo "Finished cleasing $split"
done && \

echo "Finished Cleaning" && \

mkdir bpe && \

python subword_nmt/subword_nmt/learn_joint_bpe_and_vocab.py \
-i raw/wikitext-103/wiki.train.clean.tokens \
-o bpe/wiki.bpe \
-s $1  \
--write-vocabulary bpe/wiki.vocab \
-v && \

echo "Finished Learning BPE" && \

for split in train valid test
do
  python subword_nmt/subword_nmt/apply_bpe.py \
    -i raw/wikitext-103/wiki.$split.clean.tokens \
    -c bpe/wiki.bpe \
    -o bpe/wiki.$split.bpe.tokens \
    --vocabulary bpe/wiki.vocab 
done && \

echo "Finished Applying BPE" && \

python subword_nmt/subword_nmt/build_dictionary.py \
bpe/wiki.train.bpe.tokens 
