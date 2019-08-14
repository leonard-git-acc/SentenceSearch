python ./train.py --keyedvectors_path="./data/english.bin" ^
--train_path="./data/train.json" ^
--test_path="./data/test.json" ^
--out_dir="./output" ^
--epochs="1" ^
--steps_per_epoch="10000" ^
--do_train=False ^
--saved_model_path="./output/sentsearch_nn_1565766432.model"
pause