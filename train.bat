python ./train.py --train_path="./data/train.json" ^
--test_path="./data/test.json" ^
--out_dir="./output" ^
--saved_model_path="./output/sentsearch_gru_1567070238.model" ^
--epochs=40 ^
--batch_size=64 ^
--batches_per_epoch=1500 ^
--do_train=True ^
--do_test=False
pause