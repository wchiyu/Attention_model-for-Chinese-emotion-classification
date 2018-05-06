from flask import Flask
from flask import render_template
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
import jieba
import numpy as np
from keras.models import load_model
import h5py

class NameForm(FlaskForm):
    name=StringField('Please input your comment', validators=[DataRequired()])
    submit=SubmitField('submit')
    
app = Flask(__name__)
bootstrap=Bootstrap(app)

app.config['SECRET_KEY'] = 'hhhhhh'
app.config['BOOTSTRAP_SERVE_LOCAL']=True

print('loading the model.........')
file_path = 'LSTM_test_4.h5'
model = load_model(file_path)
f = open('dic.txt', 'r')
a = f.read()
dic = eval(a)
f.close()
print('model has loaded')

@app.route('/',methods=['GET','POST'])
def index():
    name=None
    form = NameForm()
    result=0.0
    str_result=None
    if form.validate_on_submit():
        name = form.name.data
        str_numpy=str2number(name,100)
        s0_train = np.zeros((1, 100))
        c0_train = np.zeros((1, 100))
        result = model.predict([str_numpy, s0_train, c0_train])
        result=float(result)
        if result>=0.5:
            str_result='positive'
        else:
            str_result='negative'
        form.name.data = ''
    return render_template('index.html', name=result,str_result=str_result,form1=form)

def str2number(data,max_len=100):

    tmp=[]
    seg = jieba.cut(data, cut_all=False, HMM=True)
    for word in seg:
        if word in dic:
            tmp.append(dic[word])
        else:
            tmp.append(0.0)
    for i in range(max_len-len(tmp)):
        tmp.append(0.0)
    str_numpy=np.array(tmp)
    str_numpy=str_numpy.astype(np.int64)
    str_numpy=str_numpy.reshape((1,-1))
    return str_numpy

if __name__=='__main__':
    app.run(debug=False)

