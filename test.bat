python ./train.py --train_path="./data/train.json" ^
--test_path="./data/test.json" ^
--out_dir="./output" ^
--epochs="1" ^
--batches_per_epoch="350" ^
--saved_model_path="./output/sentsearch_gru_1567070238.model" ^
--do_train=False ^
--do_test=True
pause