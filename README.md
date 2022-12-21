# KE-T5 CommonGen Finetuning
KE-T5 사전학습된 모델을 CommonGen task에 맞게 finetuning

* 참고레포
    * https://github.com/nlpai-lab/KommonGen
    * https://github.com/MichaelZhouwang/Commonsense_Pretraining 

## Installation
```
git clone https://github.com/cozy-ai/ke-t5-kommongen.git
cd ke-t5-kommongen
conda create -n t5-kg
conda activate t5-kg
pip3 install -r requirements.txt
```

## Data Preprocessing
finetuning에는 KommonGen(https://github.com/nlpai-lab/KommonGen) 데이터를 사용하였다. 이 데이터는 /datasets/raw/ 폴더에서 확인할 수 있다.
해당 데이터를 T5 seq2seq task format에 맞추기 위하여 먼저 전처리 과정을 수행해야 한다. `preprocessing.py`을 실행하면 /datasets/final/ 폴더에 최종 데이터가 생성된다.
```
python3 preprocessing.py
```

## Training
```
python3 run_experiment.py
```