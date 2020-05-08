#-*-coding:utf-8-*-
import re
import time
import numpy as np
import random
from collections import defaultdict
import scipy.sparse as sp
import bloscpack as bp

from extract_fr_tense import *

class DepGraph:

    ROOT_TOKEN = '<root>'
    
    def __init__(self,sent_id,deprel,feats,edges,wordlist=None,upos_tags=None,xpos_tags=None,with_root=False):
         
        self.gov2dep = { }
        self.has_gov = set()            #set of nodes with a governor
        self.root_id = None
        self.sentence_id = sent_id


        for (gov,label,dep) in edges:
            self.add_arc(gov,label,dep)
 
        if with_root:
            self.add_root()

        if wordlist is None:
            wordlist = []
        self.words    = [DepGraph.ROOT_TOKEN] + wordlist 
        self.upos_tags = [DepGraph.ROOT_TOKEN] + upos_tags if upos_tags else None
        self.xpos_tags = [DepGraph.ROOT_TOKEN] + xpos_tags if xpos_tags else None
        self.feats_aspect    = [DepGraph.ROOT_TOKEN] + feats
        self.deprel = [DepGraph.ROOT_TOKEN] + deprel
        
        

    def add_root(self):
            
        if self.gov2dep and 0 not in self.gov2dep:
            root = list(set(self.gov2dep) - self.has_gov)
            if len(root) == 1:
                self.add_arc(0,'root',root[0])
                self.root_id = root[0]
            else:
                assert(False) #no single root... problem.
        elif not self.gov2dep: #single word sentence
            self.add_arc(0,'root',1)
                
    def add_arc(self,gov,label,dep):
        """
        Adds an arc to the dep graph
        """
        if gov in self.gov2dep:
            self.gov2dep[gov].append( (gov,label,dep) )
        else:
            self.gov2dep[gov] = [(gov,label,dep)]
            
        self.has_gov.add(dep)

    @staticmethod
    def read_tree(istream):
        """
        Reads a conll tree from input stream 
        """
        def graph(conll,sent_id):
            words   = [ ]
            upostags = [ ]
            xpostags = [ ]
            edges   = [ ]
            feats_aspect   = [ ]
            deprel = [ ]
            for dataline in conll:
                if '-' in dataline[0]:
                    continue          #skips compound word annotation
                words.append(dataline[1])
                upostags.append(dataline[3])
                xpostags.append(dataline[4])
                deprel.append(dataline[7].strip())
                if dataline[6] != '0': #do not add root immediately
                    edges.append((int(dataline[6]),dataline[7].strip(),int(dataline[0]))) # shift indexes !
                
                if dataline[5] != '_' and 'Aspect' in dataline[5]: # éliminer les verbes participe passé
                    aspect= dataline[5].split('=')[1]
                    #feat_tense = [x for x in feats_list if 'Tense' in x] 
                    feats_aspect.append(aspect)
                else:
                    feats_aspect.append('_')                   
            return DepGraph(sent_id,deprel,feats_aspect,edges,words,upos_tags=upostags,xpos_tags=xpostags,with_root=True)

        conll = [ ]
        deptrees = [ ]
        line  = istream.readline( ) # a string
        #checks whether the string consists of whitespace or contains #
        while istream and line.startswith('#'):#(line.isspace() or ):
            if line.startswith('# sentenceID'):
                sent_id = line.split("=")[1].strip()
                sent_id = sent_id.split('_')
                #print("#1",sent_id)
                if len(sent_id[1])==1:
                    sent_id = int(sent_id[0]+'0'+sent_id[1])
                else:
                    sent_id = int(sent_id[0]+sent_id[1])                        
            line  = istream.readline()

        while istream and not line.strip() == '':
            line = line.split('\t') #split this String by tabulator into an array
            conll.append(line)
            line = istream.readline()
            while line.startswith('#'):
                if line.startswith('# sentenceID'):
                    deptree = graph(conll,sent_id)
                    deptrees.append(deptree)
                    conll = [ ]
                    sent_id = line.split("=")[1].strip()
                    sent_id = sent_id.split('_')
                    #print("#2",sent_id)
                    if len(sent_id[1])==1:
                        sent_id = int(sent_id[0]+'0'+sent_id[1])
                    else:
                        sent_id = int(sent_id[0]+sent_id[1])                 
                line = istream.readline()

        if not conll:
        	return None
        deptrees.append(graph(conll,sent_id))
        return deptrees

    def __str__(self):
        """
        Conll string for the dep tree
        """
        lines    = [ ]
        revdeps  = dict([( dep, (label,gov) ) for node in self.gov2dep for (gov,label,dep) in self.gov2dep[node] ])
        for node in range( 1, len(self.words)  ):
            L    = ['-']*11
            L[0] = str(node)
            L[1] = self.words[node]
            if self.upos_tags:
                L[3] = self.upos_tags[node] 
            if self.xpos_tags:
                L[4] = self.xpos_tags[node]              
            label,head = revdeps[node] if node in revdeps else ('root', 0)
            if self.feats_aspect:
                L[5] = self.feats_aspect[node]               
            if self.deprel:
                L[7] = self.deprel[node]
            L[6] = str(head)
            lines.append( '\t'.join(L)) 
        return '\n'.join(lines)

    def __len__(self):
        return len(self.words)

def treebank(filename):
    istream = open(filename)
    tlist = []
    deptrees = DepGraph.read_tree(istream)
    while deptrees :
        tlist.append(deptrees)
        deptrees = DepGraph.read_tree(istream)
    istream.close()
    return tlist


#nc_zh_treebank = treebank('zh_sample_nc_v15.conll')#('../data/zh_nc_v15.conll')
nc_zh_treebank = treebank('../data/filtered_zh_sample_nc_v15.conll')
#print(len(nc_zh_treebank))
"""
for trees in nc_zh_treebank[:1]:
    for tree in trees: 
        print('#tree: \n',tree)
        print('#tree.words: ',tree.words)
        print('#tree.xpos_tags: ',tree.xpos_tags)
"""


n_docs = len(nc_zh_treebank)
docs_idx = list(range(n_docs))
random.shuffle(docs_idx)
train_len = round(n_docs*0.8)
dev_len = round(n_docs*0.1)

train_idx = docs_idx[:train_len]
dev_idx = docs_idx[train_len:train_len+dev_len]
test_idx =  docs_idx[train_len+dev_len:]

train_treebank = [nc_zh_treebank[idx] for idx in train_idx]
dev_treebank = [nc_zh_treebank[idx] for idx in dev_idx]
test_treebank = [nc_zh_treebank[idx] for idx in test_idx]
t_train_samples = 0
for doc in train_treebank:
	t_train_samples +=len(doc)
t_dev_samples = 0
for doc in dev_treebank:
	t_dev_samples +=len(doc)
t_test_samples = 0
for doc in test_treebank:
	t_test_samples +=len(doc)

n_train_samples = t_train_samples - len(train_treebank)
n_dev_samples = t_dev_samples - len(dev_treebank)
n_test_samples = t_test_samples - len(test_treebank)



#n_trainDoc = len(train_idx)
#n_testDoc = len(test_idx)
#dev_len = round(n_docs*0.1)
#train_idx, dev_idx,test_idx = docs_idx[:train_len],docs_idx[train_len:train_len+dev_len],docs_idx[train_len+dev_len:]
#print(train_idx)
#print(dev_idx)
#print(test_idx)


feats_md = ['可能', '能', '会', '未能', '必须', '可以', '应当', '足以', '不会', '不能', '需要', '应该', '应', '能否', '能够', '要', '想', '可', '不可能', '才能', '不该', '不必', '尽可能', '不难', '并不会', '只能', '不愿', '不可', '不要', '愿意', '需', '未必', '不得不', '请', '不只', '不想','得','必需','必','必定','必将','不曾']
md2idx = dict(zip(feats_md,range(len(feats_md))))
feats_tmod = ['事实上','实际上','如今','现','现在','现今','目前','当前','眼下','同时','同期','与此同时','前','从前','此前','之前','以前','不久前','最终','最后','今天','当今','今年','迄今','时至今日','今时','今日','明天','明年','后天','后年','昨天','去年','每年','每天','每月','每个星期','当时','当年','那时','当初','最初','过去','时','间','时候','期间','时期','后','其后','之后','事后','今后','此后','以后','战后','来','近年来','近几年来','近几年','近期','最近','长期以来','后来','未来','届时','永远','不久','此时','此刻','自此','本周','本月','以往', '假以时日','上个月','下个月','这次','上次','下次','刚才']
tmod2idx = dict(zip(feats_tmod,range(len(feats_tmod))))
feats_adv = ['已经','因此','正在','一直','在','仍然','仍','已','也许','正','依然','未','再','曾','往往','这就','不再','通常','常常','曾经','诚然','很快','不断','经常''日益','总是','很少','没','从来','早已','必然','即将','永远','尚未','刚刚','绝不','多年来','纷纷','至今','一般','立刻','依旧','实际上''尚','早','毫不','一再','立即','随之','马上','原本','就要','从未','总','而后','始终','本来','常','暂时','必将','后','最近','向来','近','时常','多年''众所周知','大都','从不','将要','快','终将','从此','也曾','未曾' ]
advmod2idx = dict(zip(feats_adv,range(len(feats_adv))))
feats_temps = ['UNK','Past','Pres','Fut']
temps2idx = dict(zip(feats_temps,range(len(feats_temps))))
feats_aspect = ['了','过', '着']
mark2idx = dict(zip(feats_aspect,range(len(feats_aspect))))


valence_mark = ['将','把','被']

def make_w2idx(train_idx):
	wordset = set([])
	WPset = set([])
	punct =  "[.!//_,$&%^*()<>+\"'?@#-|:~{}]+|[——！\\\\，。=？、：“”‘’《》【】￥……（）]+"
	for doc_idx in train_idx:
		for sent_tree in nc_zh_treebank[doc_idx]:			
			len_s = len(sent_tree.words)
			for node in range(1,len_s):
				word = sent_tree.words[node]
				xpos = sent_tree.xpos_tags[node]
				upos = sent_tree.upos_tags[node]
				if upos != "PUNCT" and (not bool(re.search(punct,word))) and (not bool(re.search('[a-zA-Z]', word))): #	xpos != 'FW': #		
					wordset.add(word)
					if xpos == "FW":
						print(word)	
					WPset.add((word,xpos))
	return wordset,WPset

wordset,WPset=make_w2idx(train_idx)
wordset = list(wordset)
WPset = list(WPset)
word2idx = dict(zip(wordset,range(len(wordset))))
WP2idx = dict(zip(WPset,range(len(WPset))))
print("words set length: ",len(wordset))
print("WP set length: ",len(WPset))

n_features = len(feats_temps)+len(wordset)+len(feats_aspect)+len(feats_tmod)+len(feats_adv)+len(feats_md)+len(WP2idx)
#n_train_samples = len(sent2tense)-n_docs

print("#features: ",n_features)
print("#train_samples: ",n_train_samples)
print("#dev_samples: ",n_test_samples)
print("#test_samples: ",n_test_samples)

def oneHotEncoding(input,w2id_dico):
	
	if input:
		#integer_encoded = [feat2id_dico[word] for word in input]
		#feats = [0 for _ in range(len(w2id_dico))]
		feats = np.zeros(len(w2id_dico))#,dtype=np.int)
		for word in input:
			if word in w2id_dico: #manages unk words (ignore)
				feats[w2id_dico[word]] = 1
			#else:
			#	feats = np.zeros(len(w2id_dico)) # aspect_mark raise error here 		
		#[0 for _ in range(len(w2id_dico))]
	else:
		feats = np.zeros(len(w2id_dico))#,dtype=np.int)
		#feats = [0 for _ in range(len(w2id_dico))]
	return feats


def tree2features(deptree):

	#feature n°1 context tense: the major tense of the precedent sentence
	sentID = deptree.sentence_id
	# we don't analyse the 1st sentence (title) of each doc, so the contexte of the 1st sent after title is 'UNK'
	context_tense = [sent2tense[sentID-1]]
	ylabel = temps2idx[sent2tense[sentID]]
	#print(context_tense)  
	F_context = oneHotEncoding(context_tense,temps2idx)
	#print(F_context)
	#print(F_context.shape)
	
	aspect_mark = set([])
	tmod = set([])
	advmod = set([])
	md = set([])
	WP = set([])

	root_id = deptree.root_id
	if not root_id:
		print('sentenceID=',deptree.sentence_id)
		print('****no root:\n',deptree)
		print('**** words: ',deptree.words)
		print('**** xpos: ',deptree.xpos_tags) 
		F_VV = oneHotEncoding([deptrees.words[1]],word2idx)
		F_aspect = oneHotEncoding(aspect_mark,mark2idx) 
		F_tmod = oneHotEncoding(tmod,tmod2idx)
		F_advmod = oneHotEncoding(advmod,advmod2idx)
		F_md = oneHotEncoding(md,md2idx)
		F_WP = oneHotEncoding([(deptrees.words[1],deptrees.xpos_tags[1])],WP2idx)      
				
	else:
		root_dep = deptree.gov2dep[root_id]# pas sûr d'avoir toujours root-dep? 
		root_deprel = [edge[1] for edge in root_dep]

		#feature n°2: major verb VV of a sentence
		# -if root is a verb or if non-verb root don't have AUX dependent, then VV==root_word
		# -if root is not a verb and AUX dependent exist, 
		# if AUX aux (/MD) exist, VV = premier MD_word; elif AUX cop exist, VV=cop_word; 
		# else VV=root_word
		if deptree.upos_tags[root_id]=="VERB": 
			VV_id = root_id
		elif "aux" in root_deprel:
			edge_aux = [x for x in root_dep if x[1]== "aux" and deptree.upos_tags[x[2]]=="AUX" ]
			if edge_aux:
				VV_id = edge_aux[0][2]
			else:
				VV_id = root_id
		elif "cop" in root_deprel and deptree.upos_tags[root_id]!="ADJ":
			edge_cop = [x for x in root_dep if x[1]== "cop" and deptree.upos_tags[x[2]]=="AUX"]
			if edge_cop:
				VV_id = edge_cop[0][2]
			else:
				VV_id = root_id
		else:
			VV_id = root_id
		VV = [deptree.words[VV_id]]
		F_VV = oneHotEncoding(VV,word2idx)
		#inverted = argmax(np.array(F_VV))
		#print("VV: ",VV)
		#print("VV_onehot: ",inverted)
		#print(wordset[681])

		# feature n°3 aspect marker
		if 'case:aspect' in root_deprel:
			aspect_edge = [x for x in root_dep if x[1]=="case:aspect"]
			aspect_mark.add(deptree.words[aspect_edge[0][2]])
		F_aspect = oneHotEncoding(aspect_mark,mark2idx)


		# feature n°4 nominal tense modifier	
		if "nmod:tmod" in root_deprel:
			edge_tmod = [x for x in root_dep if x[1]=="nmod:tmod"]
			for edge in edge_tmod:
				if deptree.words[edge[2]] in feats_tmod:
					tmod.add(deptree.words[edge[2]])
		F_tmod = oneHotEncoding(tmod,tmod2idx)
		
		# feature n°5 nominal tense modifier
		if "advmod" in root_deprel:
			edge_advmod = [x for x in root_dep if x[1]=="advmod"]
			for edge in edge_advmod:
				if deptree.words[edge[2]] in feats_adv:
					advmod.add(deptree.words[edge[2]])
		F_advmod = oneHotEncoding(advmod,advmod2idx)

		# feature n°6 modal auxiliaire
		if "aux" in root_deprel:
			edge_aux = [x for x in root_dep if x[1]== "aux" and deptree.xpos_tags[x[2]]=="MD" ]
			if edge_aux:
				for edge in edge_aux:
					if deptree.words[edge[2]] in feats_md:
						md.add(deptree.words[edge[2]])
		F_md = oneHotEncoding(md,md2idx)

		# feature n°7 word/POS
		for node in range(1,len(deptree.words)):
			WP.add((deptree.words[node],deptree.xpos_tags[node]))
		F_WP = oneHotEncoding(WP,WP2idx)

		"""
		# feature passif and disposal structures
		if "aux:pass" in root_deprel:
			passif_edge = [x for x in root_dep if x[1]=="aux:pass"]
			valence_mark = deptree.words[passif_edge[0][2]]
		elif "obl:patient" in root_deprel:
			patient_edge = [x for x in root_dep if x[1]=="obl:patient"]
			patient_id = patient_edge[0][2]
			if patient_id in deptree.gov2dep[patient_id]:
				#dispo_edge = [x for x in root_dep if x[1]=="obl:patient"]
				if (patient_id,"case",deptree.words.index('把')) in deptree.gov2dep[patient_id]:
					valence_mark = '把'
				elif (patient_id,"case",deptree.words.index('将')) in deptree.gov2dep[patient_id]:
					valence_mark = '将'
		"""	
		#deptree.words[VV_id]							
		

	x1 = np.concatenate((F_context,F_aspect,F_VV))
	x2 = np.concatenate((x1,F_tmod,F_advmod))
	xfeatures = np.concatenate((x2,F_md,F_WP))

	return xfeatures.reshape(1,-1),ylabel



def make_samples(dataset,n_samples,n_features):
	
	X_matrix=np.zeros((n_samples,n_features))
	Y = []
	id_sample = 0	 
	for doc_idx in dataset:
		for deptree in nc_zh_treebank[doc_idx]:
			sentID = deptree.sentence_id
			if len(str(sentID)) < 3:
				sentID_str = '00' + str(sentID)
			else:
				sentID_str = str(sentID)
			# ignore the first sentence (the title) of every document
			if sentID_str[-2:]=="00":
				continue				 
			else:
				xfeatures,y = tree2features(deptree)
				X_matrix[id_sample]=xfeatures
				id_sample += 1
				Y.append(y)
	return X_matrix,np.array(Y)

X_train,y_train = make_samples(train_idx,n_train_samples,n_features)
X_dev,y_dev = make_samples(dev_idx,n_dev_samples,n_features)
X_test,y_test = make_samples(test_idx,n_test_samples,n_features)

tsizeMB = sum(i.size*i.itemsize for i in (X_train,X_dev,X_test))/2**20.
#blosc_args = bp.DEFAULT_BLOSC_ARGS
#blosc_args['clevel'] = 6
t = time.time()
bp.pack_ndarray_to_file(X_train, '../data/X_train.blp')#, blosc_args=blosc_args)
bp.pack_ndarray_to_file(y_train, '../data/y_train.blp')
bp.pack_ndarray_to_file(X_dev, '../data/X_dev.blp')#, blosc_args=blosc_args)
bp.pack_ndarray_to_file(y_dev, '../data/y_dev.blp')
bp.pack_ndarray_to_file(X_test, '../data/X_test.blp')#, blosc_args=blosc_args)
bp.pack_ndarray_to_file(y_test, '../data/y_test.blp')
print(y_test)
print(type(y_test))
print(y_test.shape)

t1 = time.time() - t
print("store time = %.2f (%.2f MB/s)" % (t1, tsizeMB/t1))


"""
print(X_train.shape)
print(len(y_train))
print(X_test.shape)
print(len(y_test))
#Sxfeatures = sp.csr_matrix(X_matrix)
#print(Sxfeatures)
#print(sp.csr_matrix(X_matrix[1]))
"""
