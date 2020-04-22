#-*-coding:utf-8-*-
import stanza
import pkuseg

stanza.download('fr')
stanza.download('zh-hans')
nlp_fr = stanza.Pipeline('fr',tokenize_no_ssplit=True)
nlp_zh = stanza.Pipeline('zh-hans',tokenize_pretokenized=True,tokenize_no_ssplit=True)
pkuTokenizer = pkuseg.pkuseg(model_name='news')

import time 
s1 = time.perf_counter()
file  = open('../data/news-commentary-v15.fr-zh.tsv','r')
fr_write = open('../data/fr_nc_v15.conll','w')
zh_write = open('../data/zh_nc_v15.conll','w')

# nettoyer le corpus et le passe au parser 
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
                    seg_fr = line[0].replace('ampquot;','"')
                    seg_zh = line[1]
                    doc_fr.append(nlp_fr(seg_fr))
                    seg_zh_pretokenize = pkuTokenizer.cut(seg_zh) # une liste de string
                    doc_zh.append(nlp_zh([seg_zh_pretokenize]))       
        else:
            
            conll_fr.append(doc_fr)
            conll_zh.append(doc_zh)
            doc_fr = [ ]
            doc_zh = [ ]
        
    return conll_fr,conll_zh

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
		writer.write('\n') # ligne vide: frontière doc

write_conll(conll_fr,fr_write)
write_conll(conll_zh,zh_write)

file.close()
fr_write.close()
zh_write.close()
e1 = time.perf_counter()
print ("#######################tsv2conll:",e1-s1)
