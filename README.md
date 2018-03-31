# Attention_model-for-Chinese-emotion-classification
this is a soft attention model for Chinese emotion classification
In the dir process_data,you can read file follow read_txt.py->mkdic.py->word_number.py.
In the dir model_LSTM,first I create four models,simple LSTM->biLSTM->biLSTM and LSTM->biLSTM with attention model.
The length of each sentence is 200,if you run the model you will find only attention_model do well,but if you reduce the length to 100,you will find all models preform well.
note：Training of models takes a long time，please ensure that not too many iterations，unless your computer performs very well
