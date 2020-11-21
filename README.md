# dl_optimizers

### Repository Structures:

```
dl_optimizers/
├── optimizers/
│   ├── adam/
│   ├── lars/
│   ├── lamb/
│   ├── sgd/
├── qa_system/
│   ├── data/
│   ├── results/
|   ├── machine_learning/    
│   ├── preprocess.py
│   ├── DrQA.py
├── image_classification/
│   ├── results/
│   ├── image_classification.py
|   evaluation.py
```

### Requirements:
- Python 3.7
- numpy 
- pandas
- scikit-learn
- pytorch
- torchtext
- torchvision
- spacy

### How to run:

1) Images Classification: 
```
cd image_classification/ 
python image_classification.py
```

2) QA System:

```
cd qa_system/
python preprocess.py
python DrQA.py --batch_size 64 --optimizer sgd
```

### License
MIT
