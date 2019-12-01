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
flags.DEFINE_float('lr', 1, 'the learning rate')
flags.DEFINE_float('lr_decay', 0.95, 'the learning rate decay')
flags.DEFINE_integer('emdedding_dim', 300, 'embedding dim')  
flags.DEFINE_integer('hidden_neural_size', 300, 'LSTM hidden neural size')
flags.DEFINE_integer('valid_num', 100, 'epoch num of validation')
flags.DEFINE_integer('checkpoint_num', 1000, 'epoch num of checkpoint')
flags.DEFINE_float('init_scale', 0.1, 'init scale')
flags.DEFINE_float('keep_prob', 0.2, 'dropout rate')
flags.DEFINE_integer('num_epoch', 30000, 'num epoch')
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
def get_data(word_vec_model,embed_dim):
	words=[]
	resu1t=[]
	with open("questions_2.txt",encoding="utf-8") as file_object:
		contents=re.split(' |\n',file_object.read())
		for each in contents:
			words.append(each)
	leng=len(words)
	vec=[]
	#vec=np.zeros([leng,embed_dim],dtype=float)	
	for i in range(leng):
		try:
			vec.append(word_vec_model[words[i]])
		except:
			#给一个默认值或者其他的解决办法类似于n gram的
			#可以跳过未出现的值
			pass
	vec=np.array(vec)
	return vec
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
			sumi1=tf.reduce_mean(self.cell_outputs, axis=1) #把n的单词的结果合并
			#diff=tf.square(tf.subtract(sumi1,self.target), name='err_l1')
			#diff = tf.reduce_mean(diff, axis=1)
			#self.loss=tf.clip_by_value(tf.nn.sigmoid(diff),1e-7,1.0-1e-7)
			self.loss=1-consine_similarity(sumi1,self.target)
		with tf.name_scope('cost'):
			self.cost = tf.reduce_mean(self.loss)
			#self.cost=1-consine_similarity(sumi1,self.target)
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
		self.new_lr = tf.compat.v1.placeholder(tf.float32, shape=[], name="new_learning_rate")
		self._lr_update = tf.compat.v1.assign(self.lr, self.new_lr)
	def assign_new_lr(self, session, lr_value):
		lr, _ = session.run([self.lr, self._lr_update], feed_dict={self.new_lr: lr_value})
	def assign_new_batch_size(self, session, batch_size_value):
		session.run(self._batch_size_update, feed_dict={self.new_batch_size: batch_size_value})
def evaluate(model, session, vec_data,pre_target):
	#fetches = [model.cost,model.ac_num]
	fetches = [model.cost]
	feed_dict = {}
	feed_dict[model.input_data]=vec_data
	feed_dict[model.target]=pre_target
	feed_dict[model.is_learning]=False
	model.assign_new_batch_size(session, len(vec_data))
	#cost,ac_num= session.run(fetches, feed_dict)
	cost= session.run(fetches, feed_dict)
	return cost
def evaluate_all(LEN,model, session,data):
	all_cost=0
	count=0
	for step,(vec_data,pre_target) in enumerate(batch_iter(LEN,data, batch_size=FLAGS.batch_size)):
		#fetches = [model.cost,model.ac_num]
		fetches = [model.cost]
		feed_dict = {}
		feed_dict[model.input_data]=vec_data
		feed_dict[model.target]=pre_target
		feed_dict[model.is_learning]=False
		model.assign_new_batch_size(session, len(vec_data))
		#cost,ac_num= session.run(fetches, feed_dict)
		cost= session.run(fetches, feed_dict)
		val=cost[0]
		all_cost+=val
		count+=1
		if count==100:
			break
	return all_cost/count
def batch_iter(LEN,data, batch_size):
	vec_data=[]
	pre_target=[]
	temp=[]
	lab=0
	for each in data:
		if lab==1:
			pre_target.append(each)
		temp.append(each)
		if len(temp)>LEN:
			temp.pop(0)
			lab=1
			vec_data.append(copy.deepcopy(temp))
	vec_data.pop(-1)
	vec_data = np.array(vec_data)
	pre_target= np.array(pre_target)
	data_size = len(vec_data)
	num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
	for batch_index in range(num_batches_per_epoch):
		start_index = batch_index * batch_size
		end_index = min((batch_index + 1) * batch_size, data_size)
		return_vec_data = vec_data[start_index:end_index]
		return_pre_target = pre_target[start_index:end_index]
		yield [return_vec_data,return_pre_target]
def run_epoch(LEN,train_model,session,data,global_steps):
	for step,(vec_data,pre_target) in enumerate(batch_iter(LEN,data, batch_size=FLAGS.batch_size)):
		feed_dict={}
		feed_dict[train_model.input_data]=vec_data
		feed_dict[train_model.target]=pre_target
		feed_dict[train_model.is_learning]=True
		train_model.assign_new_batch_size(session, len(vec_data))
		#fetches = [train_model.cost,train_model.ac_num,train_model.train_op]
		fetches = [train_model.cell_outputs,train_model.target,train_model.cost,train_model.train_op]
		#cost,ac_num,_= session.run(fetches, feed_dict)
		cell_outputs,target,cost,_= session.run(fetches, feed_dict)
		if (global_steps % 500 == 0):
			print('cost')
			print(cost)
			valid_cost=evaluate_all(LEN,train_model, session, data)
			print("the "+str(global_steps)+" step,valid cost is "+str(valid_cost))
		global_steps+=1
	return global_steps
def train_step(LEN):
	config = Config()
	eval_config = Config()
	print('loading word_vec model')
	word_vec_model=load_model('sgns.wiki.word')
	print('word_vec model loaded')
	gpu_config=tf.compat.v1.ConfigProto()
	gpu_config.gpu_options.allow_growth=True
	with tf.Graph().as_default(), tf.compat.v1.Session() as session:
	#with tf.compat.v1.Session() as session:
		#loadname='my-model-0'
		#with open(loadname,'wb') as f:
		#	saver=tf.compat.v1.train.Saver()
		#	saver.restore(sess,loadname)
		initializer = tf.random_normal_initializer(0.0, 0.2, dtype=tf.float32)
		#initializer=tf.compat.v1.global_variables_initializer()
		with tf.compat.v1.variable_scope("model", reuse=None, initializer=initializer):
			train_model = LSTMRNN(LEN=LEN,config=config, sess=session, is_training=True)
		# add checkpoint
		checkpoint_dir = os.path.abspath(os.path.join(config.out_dir, "checkpoints"))
		checkpoint_prefix = os.path.join(checkpoint_dir, "model")
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
		#tf.compat.v1.global_variables_initializer.run()
		tf.initialize_all_variables().run()
		global_steps = 1
		begin_time = int(time.time())
		print("loading the dataset")
		train_data=get_data(word_vec_model,embed_dim=FLAGS.emdedding_dim)
		print("dataset loaded")
		print("begin training")
		#for i in range(config.num_epoch):
		for i in range(100000):
			#print("the %d epoch training..." % (i + 1))
			train_model.assign_new_lr(session, config.lr)
			#for each epoch we reset the learning rate
			global_steps = run_epoch(LEN,train_model, session, train_data, global_steps) 
			#if i % config.checkpoint_every == 0:
			#path = saver.save(session, checkpoint_prefix, global_steps)
			if i%50==0:
				saver.save(session, 'my-model', global_step=global_steps)
				print("Saved model at i=",i)
		print("train finished")
		end_time = int(time.time())
		print("training takes %d seconds\n" % (end_time - begin_time))
if __name__ == "__main__":
	tf.compat.v1.app.run(train_step(2))
