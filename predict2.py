import pandas as pd
import numpy as np
import tensorflow as tf
import gensim
import time
import random
import string
import re
import copy
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 32, 'the batch_size of the training procedure')
flags.DEFINE_float('lr', 0.01, 'the learning rate')
flags.DEFINE_float('lr_decay', 0.95, 'the learning rate decay')
flags.DEFINE_integer('emdedding_dim', 300, 'embedding dim')  
flags.DEFINE_integer('hidden_neural_size', 300, 'LSTM hidden neural size')
flags.DEFINE_integer('valid_num', 100, 'epoch num of validation')
flags.DEFINE_integer('checkpoint_num', 1000, 'epoch num of checkpoint')
flags.DEFINE_float('init_scale', 0.1, 'init scale')
flags.DEFINE_float('keep_prob', 0.2, 'dropout rate')
flags.DEFINE_integer('num_epoch', 360, 'num epoch')
flags.DEFINE_integer('max_decay_epoch', 100, 'num epoch')
flags.DEFINE_integer('max_grad_norm', 5, 'max_grad_norm')
flags.DEFINE_string('out_dir', os.path.abspath(os.path.join(os.path.curdir, "runs_new1")), 'output directory')
flags.DEFINE_integer('check_point_every', 20, 'checkpoint every num epoch ')
def consine_similarity(x1, x2):
	x1_norm = tf.sqrt(tf.reduce_sum(tf.square(x1), axis=1))
	x2_norm = tf.sqrt(tf.reduce_sum(tf.square(x2), axis=1))
	# 内积
	x1_x2 = tf.reduce_sum(tf.multiply(x1, x2), axis=1)
	cosin = x1_x2 / (x1_norm * x2_norm)
	return cosin
def consine_similarity2(x1, x2):
	x1=tf.cast(x1,tf.float32)
	x2=tf.cast(x2,tf.float32)
	x1_norm = tf.sqrt(tf.reduce_sum(tf.square(x1)))
	x2_norm = tf.sqrt(tf.reduce_sum(tf.square(x2)))
	# 内积
	x1_norm=tf.cast(x1_norm,tf.float32)
	x2_norm=tf.cast(x2_norm,tf.float32)
	x1_x2 = tf.reduce_sum(tf.multiply(x1, x2))
	x1_x2=tf.cast(x1_x2,tf.float32)
	cosin = x1_x2 / (x1_norm * x2_norm)
	return cosin	
class Config(object):
	hidden_neural_size = FLAGS.hidden_neural_size
	embed_dim = FLAGS.emdedding_dim
	keep_prob = FLAGS.keep_prob
	lr = FLAGS.lr
	lr_decay = FLAGS.lr_decay
	batch_size = FLAGS.batch_size
	max_grad_norm = FLAGS.max_grad_norm
	num_epoch = FLAGS.num_epoch
	max_decay_epoch = FLAGS.max_decay_epoch
	valid_num = FLAGS.valid_num
	out_dir = FLAGS.out_dir
	checkpoint_every = FLAGS.check_point_every
def load_model(filename):
	model = gensim.models.KeyedVectors.load_word2vec_format(filename)
	return model
class LSTMRNN(object):
	def singleRNN(self, data, scope, cell='lstm', reuse=None):
		if cell == 'gru':
			with tf.variable_scope('grucell' + scope, reuse=reuse, dtype=tf.float64):
				used_cell = tf.contrib.rnn.GRUCell(self.hidden_neural_size, reuse=tf.get_variable_scope().reuse)
		else:
			with tf.compat.v1.variable_scope('lstmcell' + scope, reuse=reuse, dtype=tf.float64):
				used_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_neural_size, forget_bias=1.0, state_is_tuple=True,reuse=tf.compat.v1.get_variable_scope().reuse)
		with tf.name_scope('RNN_' + scope), tf.compat.v1.variable_scope('RNN_' + scope, dtype=tf.float64):
			######outs, _ = tf.nn.dynamic_rnn(used_cell, x, initial_state=self.cell_init_state, time_major=False,dtype=tf.float64)
			outs, _ = tf.nn.dynamic_rnn(used_cell, data, time_major=False,dtype=tf.float64)
			#Time_major决定了inputs Tensor前两个dim表示的含义 time_major=False时[batch_size, sequence_length, embedding_size]time_major=True时[sequence_length, batch_size, embedding_size]
		return outs
	def __init__(self ,LEN, config, sess, is_training=True):
		self.keep_prob=config.keep_prob
		self.batch_size=tf.Variable(0, dtype=tf.int32, trainable=False)
		embed_dim=config.embed_dim
		self.input_data=tf.compat.v1.placeholder(tf.float64, [None, LEN, embed_dim])
		self.target=tf.compat.v1.placeholder(tf.float64, [None, embed_dim])
		self.is_learning = tf.compat.v1.placeholder(tf.bool)
		self.hidden_neural_size = config.hidden_neural_size
		self.new_batch_size = tf.compat.v1.placeholder(tf.int32, shape=[], name="new_batch_size")
		self._batch_size_update = tf.compat.v1.assign(self.batch_size, self.new_batch_size)
		#self.ac_num=0
		# 使用dropout
		#if self.keep_prob < 1:
		#	self.input_data = tf.nn.dropout(self.input_data, rate=self.keep_prob)
		with tf.name_scope('lstm_output_layer'):
			self.cell_outputs= self.singleRNN(data=self.input_data, scope='side1', cell='lstm', reuse=None)
		with tf.name_scope('loss'):
			self.sumi1=tf.reduce_mean(self.cell_outputs, axis=1) #把n的单词的结果合并
			#diff=tf.square(tf.subtract(sumi1,self.target), name='err_l1')
			#diff = tf.reduce_mean(diff, axis=1)
			#self.loss=tf.clip_by_value(tf.nn.sigmoid(diff),1e-7,1.0-1e-7)
			self.loss=1-consine_similarity(self.sumi1,self.target)
		with tf.name_scope('cost'):
			self.cost = tf.reduce_mean(self.loss)
			#self.cost=1-consine_similarity(self.sumi1,self.target)
		if not is_training:
			return
		if self.is_learning==False:
			return
		self.globle_step = tf.Variable(0, name="globle_step", trainable=False)
		self.lr = tf.Variable(0.0, trainable=False)
		tvars = tf.compat.v1.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), config.max_grad_norm)
		#变量的导数
		optimizer = tf.compat.v1.train.AdadeltaOptimizer(learning_rate=self.lr, epsilon=1e-6)
		with tf.name_scope('train'):
			self.train_op = optimizer.apply_gradients(zip(grads, tvars))
		#使用梯度修剪法优化，参数是变量和变量的导数
		self.new_lr = tf.compat.v1.placeholder(tf.float64, shape=[], name="new_learning_rate")
		self._lr_update = tf.compat.v1.assign(self.lr, self.new_lr)
	def assign_new_lr(self, session, lr_value):
		lr, _ = session.run([self.lr, self._lr_update], feed_dict={self.new_lr: lr_value})
	def assign_new_batch_size(self, session, batch_size_value):
		session.run(self._batch_size_update, feed_dict={self.new_batch_size: batch_size_value})
def batch_iter(LEN,data, batch_size):
	vec_data=[]
	temp=[]
	for each in data:
		lab=0
		for each2 in each:
			if each2!=0:
				lab=1
				break
		if lab==0:
			vec_data.append(copy.deepcopy(temp))				
		else:
			temp.append(each)
			if len(temp)>LEN:
				temp.pop(0)
	vec_data = np.array(vec_data)
	#print(vec_data)
	data_size = len(vec_data)
	num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
	for batch_index in range(num_batches_per_epoch):
		start_index = batch_index * batch_size
		end_index = min((batch_index + 1) * batch_size, data_size)
		return_vec_data = vec_data[start_index:end_index]
		yield return_vec_data
def evaluate(model, session, vec_data):
	fetches = [model.sumi1]
	feed_dict = {}
	feed_dict[model.input_data]=vec_data
	feed_dict[model.is_learning]=False
	model.assign_new_batch_size(session, len(vec_data))
	sumi1= session.run(fetches, feed_dict)
	return sumi1
def get_data(word_vec_model,embed_dim):
	words=[]
	with open("questions_.txt",encoding="utf-8") as file_object:
		contents=re.split(' |\n',file_object.read())
		for each in contents:
			words.append(each)
	leng=len(words)
	vec=[]
	for i in range(leng):
		try:
			vec.append(word_vec_model[words[i]])
		except:
			if words[i]=='MASK':
				vec.append(np.zeros(embed_dim))
	vec=np.array(vec)
	return vec
def predict(LEN):
	config = Config()
	eval_config = Config()
	print('loading word_vec model')
	word_vec_model=load_model('sgns.wiki.word')
	words_detail = word_vec_model.wv.vocab
	print('word_vec model loaded')
	gpu_config=tf.compat.v1.ConfigProto()
	gpu_config.gpu_options.allow_growth=True
	with tf.Graph().as_default(), tf.compat.v1.Session() as session:
		initializer = tf.random_normal_initializer(0.0, 0.2, dtype=tf.float64)
		#initializer=tf.compat.v1.global_variables_initializer()
		with tf.compat.v1.variable_scope("model", reuse=None, initializer=initializer):
			model = LSTMRNN(LEN=LEN,config=config, sess=session, is_training=False)
		loadname='my-model-88560'
		#tf.initialize_all_variables().run()
		with open(loadname,'wb') as f:
			saver=tf.compat.v1.train.Saver()
			saver.restore(session,loadname)
		print("loading the dataset")
		test_data=get_data(word_vec_model,embed_dim=FLAGS.emdedding_dim)
		print("dataset loaded")
		vec_result=[]
		for step,(vec_data) in enumerate(batch_iter(LEN,test_data, batch_size=FLAGS.batch_size)):
			temp_vec_result=evaluate(model, session, vec_data)
			#temp_vec_result=temp_vec_result.tolist()
			temp_vec_result=temp_vec_result[0].tolist()
			vec_result+=temp_vec_result
		vec_result=np.array(vec_result)
		#print('vec_result')
		#print(vec_result)
		answer_list=[]
		answer_list_valid=[]
		with open('answer.txt','r',encoding="utf-8") as f:
			contents=re.split(' |\n',f.read())
			for each in contents:
				answer_list.append(each)
				try:
					answer_list_valid.append( (each,word_vec_model[each]))
				except:
					pass
		result=[]
		cc=0
		for each in vec_result:
			cc=cc+1
			min_dis=1000
			keep=[]
			for each2 in answer_list_valid:
				dis=1-consine_similarity2(each,each2[1]).eval()
				if dis<min_dis:
					min_dis=dis
					keep=each2[0]
			print(keep)
			result.append(keep)

		acr_num=0
		for i in range(len(answer_list)):
			if answer_list[i]==result[i]:
				acr_num+=1
		print('acr_num is',acr_num)
		print('acr_rate is',acr_num/len(answer_list))

		with open('LSTM_result_2.txt','w') as f:
			for each in result:
				f.write(str(each)+'\n')	
def main(args):
	predict(2)
if __name__=="__main__":
	tf.compat.v1.app.run()
