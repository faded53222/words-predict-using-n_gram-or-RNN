import re
words=[]
with open("answer.txt",encoding="utf-8") as f:
	contents=re.split('\n',f.read())
	for each in contents:
		words.append(each)
#print(words)
i=0
words2=[]
with open("questions_.txt",encoding="utf-8") as f:
	contents=re.split(' |\n',f.read())
	for each in contents:
		if each == 'MASK':
			words2.append(words[i])
			i=i+1
		else:
			words2.append(each)
#print(words2)
 
with open("questions_2.txt",'w',encoding='utf-8') as file_obj:
	result=' '.join(words2)
	#for each in result:
	file_obj.write(result)
