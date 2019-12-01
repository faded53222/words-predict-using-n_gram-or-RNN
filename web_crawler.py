import json
import requests
import time
from bs4 import BeautifulSoup
from lxml import etree
def get_content(url):
	body = ''
	resp = requests.get(url)
	if resp.status_code == 200:
		content = resp.text
		bs4 = BeautifulSoup(content,'lxml')
		body= bs4.find('div', class_='post_text').get_text()
	return body
web_dic={}
web_dic['http://tech.163.com/it/']=1
web_dic['http://tech.163.com/internet/']=1
web_dic['http://tech.163.com/telecom/']=1
count=0
def getNewsDetailUrlList(url):
    response = requests.get(url)
    html = response.content.decode('gbk')
    selector = etree.HTML(html)
    newsDetailList = selector.xpath('//ul[@id="news-flow-content"]//li//div[@class="titleBar clearfix"]//h3//a/@href')
    return newsDetailList
def getUrlList(baseUrl, num):
	urlList = []
	urlList.append(baseUrl)
	for i in range(2, num+1):
		urlList.append(baseUrl + "_" + str(i).zfill(2))
	return urlList
count=879
def deal():
	global count
	UrlList1=getUrlList('http://tech.163.com/special/gd2016',20)
	UrlList2=getUrlList('http://tech.163.com/special/techscience',10)
	UrlList3=getUrlList('http://tech.163.com/it',20)
	UrlList4=getUrlList('http://tech.163.com/internet',20)
	UrlList5=getUrlList('http://tech.163.com/telecom',20)
	UrlList6=getUrlList('http://tech.163.com/5g',10)
	UrlList7=getUrlList('http://tech.163.com/blockchain',10)
	UrlList=UrlList1+UrlList2+UrlList3+UrlList4+UrlList5+UrlList6+UrlList7
	for uu in UrlList:
		list=getNewsDetailUrlList(uu)
		for each in list:
			print(each)
			try:
				data=get_content(each)
			except:
				continue
			with open(str(count)+'.txt','w', encoding="utf-8") as f:
				f.write(data)
			count+=1
def deal2():
	global count
	list=getNewsDetailUrlList(uu)
	for each in list:
		print(each)
		try:
			data=get_content(each)
		except:
			continue
		with open(str(count)+'.txt','w', encoding="utf-8") as f:
			f.write(data)
		count+=1
		if count==1001:
			return
if __name__ == "__main__":
	deal()
