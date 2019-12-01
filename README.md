# words-predict-using-n_gram-or-RNN
web_crawler.py get data from website
process.py process the data using jieba fenci into words list with out words in stopwords.txt
data in /data are processed and put in /data_
questions.txt has MASK inside for you to fill with you trained model.
process_test.py process it into questions_.txt
n_gram.py train a model using data in /data_ and predict the MASK in questions_.txt get gram2_result.txt and gram3_result.txt
RNN.py train models and save,predict.py load a model and predict what the MASK is.
RNN2 and predict2.py are the same,but they use data with data in questions_2.txt which is question_.txt filled by answers with fill_question.py.
