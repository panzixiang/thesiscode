from bs4 import BeautifulSoup

soup = BeautifulSoup(open("datasets/math.txt"), "html5lib")
#printb"(soup.prettify())

#find all sub-categories
allcats = [str(x) for x in soup.find_all('categories')]
#cleanup 
allcats = [entry.replace('<categories>', '').replace('</categories>','') for entry in allcats]
#print allcats[0]

#list all abstracts
allabs = [str(x) for x in soup.find_all('abstract')]
#cleanup
allabs = [entry.replace('<abstract>', '').replace('</abstract>','').replace('\n',' ').strip() for entry in allabs]
#print len(allabs)
#print allabs[0]

withtag = map(lambda x,y:x+': '+y, allcats, allabs)
withtag.sort()
#for x in withtag[0:2]:
#    print x
#    print '\n'

print allabs[1]
