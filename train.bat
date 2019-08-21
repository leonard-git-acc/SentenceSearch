python ./train.py --keyedvectors_path="./data/english.bin" ^
--train_path="./data/train.json" ^
--test_path="./data/test.json" ^
--out_dir="./output" ^
--epochs=20 ^
--batch_size=32 ^
--batches_per_epoch=6000 ^
--do_train=True ^
--do_test=False
pause