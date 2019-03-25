
# Scene_Classification

## Submissions :

torch densenet121 - 93.56
densenet121 + sequential - 92.9
fastai stage1 - 	0.942922374429224

torch resnet152 - 0.940639269406393

torch densenet121 - only linear - 94.65 (highest)

resnet with removed 512 dense and 128, output layer - 0.942922374429224

retart resnet - not checked with AV cause val_acc is very less


## Other info

1. For using restart split the dataseet to form data-train with the directory structure like

```
data-train
- 0
- 1
- 2
- 3
- 4
- 5
```

For doing that you can try out `data_load.py` but make sure to keep a copy of the original dataset for the other models as `data_load.py` moves the file, not copy.

Once you split into the classes you don't need to split again into train and validation. My code takes care of that.

2. For some reason when you save the file to_csv the index is added so open it in excel/numbers and delete the column. Export to csv and then submit if you want. 
