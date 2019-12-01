import jieba
import re
import string
import os
def get_stopwords_list():
    stopwords = [line.strip() for line in open('stopwords.txt').readlines()]
    return stopwords
def seg_depart(sentence):
    sentence_depart = jieba.lcut(sentence.strip())
    return sentence_depart
def remove_digits(input_str):
    punc = u'0123456789%.'
    output_str = re.sub(r'[{}]+'.format(punc), '', input_str)
    return output_str
def move_stopwords(sentence_list, stopwords_list):
    out_list = []
    for word in sentence_list:
        if word not in stopwords_list:
            if not remove_digits(word):
                continue
            if word != '\t' and word!=' ' and len(word)<25:
                out_list.append(word)
    return out_list
if __name__ == "__main__":
	stopwords = get_stopwords_list()
	word_list=[]
	word_dic={}
	word_count=0
	read_num=1000
	path1='./data/'
	path2='./data_/'
	if not os.path.exists(path2):
		os.makedirs(path2) 
	for i in range(read_num):
		resu1t=[]
		with open(path1+str(i)+".txt",encoding="utf-8") as file_object:
			result=[]
			contents = file_object.read()
			lines=re.split('[。|\n|？|?]',contents)#这里可以加入逗号，作为分句的界限
			for each in lines:
				each .replace('\xa0','')
				each.replace(u'\u3000', u'')
				if len(each.strip())!=0: 
					#print(each.strip())
					seg_list = jieba.cut(each.strip(),cut_all=False)
					seg_list=move_stopwords(seg_list, stopwords)
					#print(seg_list)
					word_count+=len(seg_list)
					for each2 in seg_list:
						if each2 not in word_dic.keys():
							word_list.append(each2)
							word_dic[each2]=0
						word_dic[each2]+=1
					result.append(' '.join(seg_list))
		with open(path2+str(i)+"_.txt",'w',encoding='utf-8') as file_obj:
			for each in result:
				file_obj.write(each+'\n')
	with open("word_list.txt",'w',encoding='utf-8') as file_obj:
		for i in range(len(word_list)):
			file_obj.write(str(i)+" "+str(word_list[i])+'\n')
	#print(word_dic)
	
