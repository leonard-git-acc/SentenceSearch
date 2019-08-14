python ./train.py --keyedvectors_path="./data/english.bin" ^
--train_path="./data/train.json" ^
--test_path="./data/test.json" ^
--out_dir="./output" ^
--epochs="1" ^
--steps_per_epoch="100000" ^
--do_train=True ^
--do_test=False
pause