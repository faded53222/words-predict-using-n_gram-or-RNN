import jieba
import re
import string
import os
dic_2={}
dic_3={}
def feed(data):
	keep_2=()
	keep_3=()
	for each in data:
		if len(keep_2)==2:
			if keep_2 not in dic_2.keys():
				dic_2[keep_2]={}
			if each not in dic_2[keep_2].keys():
				dic_2[keep_2][each]=0			
			dic_2[keep_2][each]+=1
		if len(keep_3)==3:
			if keep_3 not in dic_3.keys():
				dic_3[keep_3]={}
			if each not in dic_3[keep_3].keys():
				dic_3[keep_3][each]=0
			dic_3[keep_3][each]+=1
		keep_2=list(keep_2)
		keep_3=list(keep_3)
		keep_2.append(each)
		if len(keep_2)==3:
			keep_2.pop(0)
		keep_3.append(each)
		if len(keep_3)==4:
			keep_3.pop(0)
		keep_2=tuple(keep_2)
		keep_3=tuple(keep_3)
def build():
	read_num=1000
	path='./data_/'
	for i in range(read_num):
		resu1t=[]
		with open(path+str(i)+"_.txt",encoding="utf-8") as file_object:
			contents = file_object.read()
			data=re.split('[ |\n]',contents)
			feed(data)
	path2='./data(/'
	for i in range(read_num):
		resu1t=[]
		with open(path2+str(i)+".txt",encoding="utf-8") as file_object:
			contents = file_object.read()
			data=re.split('[ |\n]',contents)
			feed(data)
	#for each in dic_2.keys():
	#	print(each,dic_2[each])
def predict(file_name):
	result_2=[]
	result_3=[]
	keep_2=()
	keep_3=()
	with open(file_name,encoding="utf-8") as file_object:
		contents = file_object.read()
		data=re.split('[ |\n]',contents)
		for each in data:
			if each!='MASK':
				keep_2=list(keep_2)
				keep_3=list(keep_3)
				keep_2.append(each)
				if len(keep_2)==3:
					keep_2.pop(0)
				keep_3.append(each)
				if len(keep_3)==4:
					keep_3.pop(0)
				keep_2=tuple(keep_2)
				keep_3=tuple(keep_3)
			else:
				print(each)
				print(keep_2,(keep_2 in dic_2.keys()))
				print(keep_3,(keep_3 in dic_3.keys()))
				when_max2='NOT_FOUND'
				when_max3='NOT_FOUND'
				if keep_2 in dic_2.keys():
					max_count2=0
					for each2 in dic_2[keep_2].keys():
						if dic_2[keep_2][each2]>max_count2:
							when_max2=each2
							max_count2=dic_2[keep_2][each2]
				result_2.append(when_max2)
				if keep_3 in dic_3.keys():
					max_count3=0
					for each3 in dic_3[keep_3].keys():
						if dic_3[keep_3][each3]>max_count3:
							when_max3=each3
							max_count3=dic_3[keep_3][each3]
				result_3.append(when_max3)
				print(when_max2)
				print(when_max3)
	with open('gram2_result.txt','w') as f:
		for each in result_2:
			f.write(str(each)+'\n')
	with open('gram3_result.txt','w') as f:
		for each in result_3:
			f.write(str(each)+'\n')	
	count2=0
	count3=0
	length=0
	with open('answer.txt',encoding="utf-8") as f:
		contents = f.read()
		data=re.split('\n',contents)
		length=len(data)
		for i in range (len(data)):
			if data[i]==result_2[i]:
				count2+=1
			if data[i]==result_3[i]:
				count3+=1				
	print('total num',length)
	print('2gram accurate num',count2)
	print('3gram accurate num',count3)
	#print(result_2)
	#print(result_3)
if __name__ == "__main__":
	build()
	predict("questions_.txt")
