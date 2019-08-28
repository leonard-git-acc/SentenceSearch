python ./train.py --train_path="./data/train.json" ^
--test_path="./data/test.json" ^
--out_dir="./output" ^
--epochs=20 ^
--batch_size=64 ^
--batches_per_epoch=3000 ^
--do_train=True ^
--do_test=False
pause