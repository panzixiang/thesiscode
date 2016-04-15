from bs4 import BeautifulSoup

soup = BeautifulSoup(open("datasets/all.xml"), "html5lib")
#printb"(soup.prettify())

#find all sub-categories
allcats = [str(x) for x in soup.find_all('categories')]
#cleanup tags 
allcats = [entry.replace('<categories>', '').replace('</categories>','') for entry in allcats]
#split individual tags and get the first as the catgory
allcatsplit = [taglist.split() for taglist in allcats]
pri_tagset = [cat[0].split('.', 1)[0]  for cat in allcatsplit]
#see uniques
uniq_tagset = sorted(set(pri_tagset))

#list all abstracts
allabs = [str(x) for x in soup.find_all('abstract')]
#cleanup tags
allabs = [entry.replace('<abstract>', '').replace('</abstract>','').replace('\n',' ').strip() for entry in allabs]

#concat for future use
withtag = map(lambda x,y:x+': '+y, allcats, allabs)
withtag.sort()
#for x in withtag[0:2]:
#    print x
#    print '\n'

print allabs[1]


#build individual lists
for i in range(len(pri_target)):
        if 'astro' in pri_target[i]: 

