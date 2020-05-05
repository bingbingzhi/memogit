#-*-coding:utf-8-*-
import stanza
import pkuseg
import re
stanza.download('fr')
# pkuseg tokenize
stanza.download('zh-hans')
nlp_fr = stanza.Pipeline('fr',tokenize_no_ssplit=True)
nlp_zh = stanza.Pipeline('zh-hans',tokenize_pretokenized=True,tokenize_no_ssplit=True)
pkuTokenizer = pkuseg.pkuseg(model_name='news')

import time 
s1 = time.perf_counter()
file  = open('../data/nc_v15.fr-zh.tsv','r')
fr_write = open('../data/filtered_fr_sample_nc_v15.conll','w')
zh_write = open('../data/filtered_zh_sample_nc_v15.conll','w')

#file  = open('t_sample_nc_v15.tsv','r')
#fr_write = open('fr_sample_nc_v15.conll','w')
#zh_write = open('zh_sample_nc_v15.conll','w')
# nettoyer le corpus et le passe au parser 


zhPattern = re.compile(u'[\u4e00-\u9fa5]+')
root_error =['比如','比如说','例如','事实上','毫无疑问','还有','尽管如此','相反','恰恰相反','鉴于','显而易见','平心而论，','总的来说，' ,'迄今为止，','说到底，','否则，','有鉴于此，','毋庸置疑，','换句话说，','相比之下，']

def doc2conll(file):
    conll_fr = [ ]
    conll_zh = [ ]
    doc_fr = [ ]
    doc_zh = [ ]
    for line in file: 
        line = line.strip()
        if line :
            if '�' not in line:
                line = line.split('\t')
                if  len(line)==2 and ('' not in line and ' 'not in line):                   
                    seg_zh = line[1]
                    if zhPattern.search(seg_zh):
                    	# remove everything between parentheses including ()
                        seg_fr = re.sub(u"\\(.*?\\)|\\（.*?\\）"," ",line[0])
                        seg_zh =re.sub(u"\\(.*?\\)|\\（.*?\\）"," ",seg_zh)
                        # all the sentence is beteween parentheses, seg==" "
                        # ignore the sentence of length < 3 words.
                        length = len(seg_fr.split())
                        if seg_fr.strip() and seg_zh.strip() and length > 3:
                            doc_fr.append(nlp_fr(seg_fr))
                            seg_zh_pretokenize = pkuTokenizer.cut(seg_zh) # une liste de string
                            # some frequent root error with stanza chinese parser,remove directly the conjunction words
                            if seg_zh_pretokenize[0] in root_error and seg_zh_pretokenize[1]=="，":
                            	print(seg_zh_pretokenize)
                            	seg_zh_pretokenize=seg_zh_pretokenize[2:]
                            	print(seg_zh_pretokenize)

                            doc_zh.append(nlp_zh([seg_zh_pretokenize]))
                    else:
                        print("*Not chinese segment: ",seg_zh)
                else:
                    print("*No chinese alignement: ",line)          
        else:
            
            conll_fr.append(doc_fr)
            conll_zh.append(doc_zh)
            doc_fr = [ ]
            doc_zh = [ ]
        
    return conll_fr,conll_zh
print("###########lecture du fichier tsv###########")
conll_fr,conll_zh = doc2conll(file)


def write_conll (conll_list, writer):
	for document in conll_list:
		idx_doc = conll_list.index(document)
		writer.writelines('# docID = '+ str(idx_doc)+'\n')
		for doc in document:
			idx_sent = document.index(doc)
			writer.writelines('# sentenceID = '+ str(idx_doc)+'_'+str(idx_sent)+'\n')
			writer.writelines('# text = '+ doc.text+'\n')
			for sentence in doc.sentences:
				for token in sentence.tokens:
					for word in token.words:
						if word.feats == None:
							feats = "_"
						else:
							feats = word.feats
						if word.xpos == None:
							xpos = "_"
						else:
							xpos = word.xpos
						writer.writelines(str(word.id)+"\t"+ word.text+"\t"+word.lemma+"\t"+word.upos+"\t"+xpos+"\t"+feats+"\t"+str(word.head)+"\t"+ str(word.deprel))
						writer.write('\n')
		writer.write('\n') # ligne vide?

print("###########écriture du fichier conll#############")
write_conll(conll_fr,fr_write)
write_conll(conll_zh,zh_write)


fr_write.close()
zh_write.close()

file.close()
e1 = time.perf_counter()
print ("#######################cube2conll:",e1-s1)

