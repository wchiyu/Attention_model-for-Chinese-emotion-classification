# Attention_model-for-Chinese-emotion-classification<br>
this is a soft attention model for Chinese emotion classification<br>
In the dir process_data,you can read file follow read_txt.py->mkdic.py->word_number.py.<br>
In the dir model_LSTM,first I create four models,simple LSTM->biLSTM->biLSTM and LSTM->biLSTM with attention model.<br>
The length of each sentence is 200,if you run the model you will find only attention_model do well,but if you reduce the length to 100,you will find all models preform well.<br>
I have provided all data for you,you can fork these and run them.<br>
If you want to test the model,you can come into the dir flask_test/ and run hello.py,the web will show in http://127.0.0.1:5000/.<br>
note：Training of models takes a long time，please ensure that not too many iterations，unless your computer performs very well<br>
