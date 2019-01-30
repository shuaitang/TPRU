mkdir nli/ && \
mkdir preprocessed/ && \
python download.py --data_dir nli --task MNLI,SNLI && \
python SNLI+MNLI_gathering.py
