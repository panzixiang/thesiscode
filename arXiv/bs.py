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

#['astro-ph', 'cond-mat', 'cs', 'gr-qc', 'hep-ex', 'hep-lat', 'hep-ph', 'hep-th', 'math', 'math-ph', 'nlin', 'nucl-ex', 'nucl-th', 'physics', 'q-bio', 'q-fin', 'quant-ph', 'stat']

#build individual lists
astro = []
cond = []
cs = []
hep = []
math = []
physics = []
qbio = []
qfin = []
quant = []
stat = []
for i in range(len(pri_tagset)):
    #print pri_tagset[i]
    if 'astro' in pri_tagset[i]:
        astro.append(allabs[i])
    elif 'cond' in pri_tagset[i]:
        cond.append(allabs[i])
    elif 'cs' in pri_tagset[i]:
        cs.append(allabs[i])
    elif 'hep' in pri_tagset[i]:
        hep.append(allabs[i])
    elif 'gr-qc' in pri_tagset[i]:
        physics.append(allabs[i])
    elif 'math' in pri_tagset[i]:
        math.append(allabs[i])
    elif 'nlin' in pri_tagset[i]:
        physics.append(allabs[i])
    elif 'nucl' in pri_tagset[i]:
        physics.append(allabs[i])
    elif 'physics' in pri_tagset[i]:
        physics.append(allabs[i])
    elif 'q-bio' in pri_tagset[i]:
        qbio.append(allabs[i])
    elif 'q-fin' in pri_tagset[i]:
        qfin.append(allabs[i])
    elif 'quant' in pri_tagset[i]:
        quant.append(allabs[i])
    elif 'stat' in pri_tagset[i]:
        stat.append(allabs[i])  
    else:
        print pri_tagset[i]
print len(astro)
print len(cond)
print len(cs)
print len(hep)
print len(math)
print len(physics)
print len(qbio)
print len(qfin)
print len(quant)
print len(stat)
