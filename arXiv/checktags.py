from bs4 import BeautifulSoup

soup = BeautifulSoup(open("datasets/all.xml"), "html5lib")
#printb"(soup.prettify())

#find all sub-categories
allcats = [str(x) for x in soup.find_all('categories')]
#cleanup tags 
allcats = [entry.replace('<categories>', '').replace('</categories>','') for entry in allcats]
#split individual tags and get the first as the catgory
allcatsplit = [taglist.split() for taglist in allcats]
pri_tagset = [cat[0]  for cat in allcatsplit]
#see uniques
uniq_tagset = sorted(set(pri_tagset))
print len(pri_tagset)
print uniq_tagset

