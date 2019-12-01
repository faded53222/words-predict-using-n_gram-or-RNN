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
	resu1t=[]
	with open("questions.txt",encoding="utf-8") as file_object:
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
				result.append(' '.join(seg_list))
	with open("questions2_.txt",'w',encoding='utf-8') as file_obj:
		for each in result:
			file_obj.write(each+'\n')
	
