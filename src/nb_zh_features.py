#-*-coding:utf-8-*-
import re
from collections import defaultdict

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
            line  = istream.readline()
        while istream and not line.strip() == '':
        	line = line.split('\t') #split this String by tabulator into an array
        	conll.append(line)
        	line = istream.readline()
        	while line.startswith('#'):
        		line = istream.readline()
        	if line[:2] == '1\t':
        		deptree = graph(conll,sent_id)
        		deptrees.append(deptree)
        		conll = [ ]
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

nc_zh_treebank = treebank('../data/zh_nc_v15.conll')

print("length of chinese treebank: ",len(nc_zh_treebank))
#print(len(nc_zh_treebank[0]))
"""
for trees in nc_zh_treebank[:1]:
    for tree in trees: 
        print('#tree: \n',tree)
        print('#tree.words: ',tree.words)     
        print('#tree.gov2dep: ',tree.gov2dep)
        print('#tree.aspect: ',tree.feats_aspect)
        print('#tree.root_id: ',tree.root_id)
        print('#tree.deprel: ',tree.deprel)
        print()

root_error = defaultdict(int)
tmod = defaultdict(int)
advmod_ADV_RB = defaultdict(int)
aux_MD = defaultdict(int)
case_ADP_BB = defaultdict(int)
for doc in nc_zh_treebank:
    for tree in doc:
        if tree.root_id == 1:
            root_error[(tree.xpos_tags[1],tree.words[1])]+=1
        idx = tree.root_id
        if not idx:
            print('sentenceID=',tree.sentence_id)
            print('****no root:\n',tree)         
            break
        
        root_dep = tree.gov2dep[idx]
        if not root_dep:
            print('sentenceID=',tree.sentence_id)
            print('****no root_dep:\n',tree)
            break
        
        def extract_words(idx,tree,root_dep,verb_root=True):
            edge_tmod = [x for x in root_dep if x[1]=="nmod:tmod"]
            if edge_tmod:
                for e in edge_tmod:
                    tmod[tree.words[int(e[2])]]+=1

            edge_advmod = [x for x in root_dep if x[1]=="advmod"]
            if edge_advmod:
                for e in edge_advmod:
                    if tree.upos_tags[int(e[2])] == "ADV" and tree.xpos_tags[int(e[2])]=="RB":
                        advmod_ADV_RB[tree.words[int(e[2])]]+=1

            edge_aux = [x for x in root_dep if x[1]=="aux"]
            if edge_aux:
                for e in edge_aux:
                    if tree.xpos_tags[int(e[2])]=="MD":
                        aux_MD[tree.words[int(e[2])]]+=1

            edge_patient = [x for x in root_dep if x[1]=="obl:patient"]
            if edge_patient:
                oblID = edge_patient[0][2]
                if oblID in tree.gov2dep:
                    dep_obl = tree.gov2dep[oblID]
                    edge_case = [x for x in dep_obl if x[1]=="case"]
                    if edge_case:
                        for e in edge_case:
                            if tree.upos_tags[int(e[2])] == "ADP" and tree.xpos_tags[int(e[2])]=="BB":
                                case_ADP_BB[tree.words[int(e[2])]]+=1
        
        if tree.upos_tags[idx]=='VERB':
            extract_words(idx,tree,root_dep)

        else:
            extract_words(idx,tree,root_dep,verb_root=False)
            edge_verb = [x for x in root_dep if tree.xpos_tags[x[2]]=="VV"]
            lowest_dist= 99
            if edge_verb:
                for e in edge_verb:
                    dist = abs(int(e[2])-idx)
                    if dist<lowest_dist:
                        lowest_dist=dist
                        verbID = int(e[2])
                if verbID in tree.gov2dep:
                    dep_v = tree.gov2dep[int(verbID)]
                    extract_words(verbID,tree,dep_v)


#print("rootID=1:\n",root_error,"\n")  
#print("nmod:tmod list:\n",tmod,"\n") 
#print("ADV_RB_advmod list:\n",advmod_ADV_RB,"\n")
#print("MD_aux list:\n",aux_MD,"\n")
#print("ADP_BB_case list:\n",case_ADP_BB,"\n")    
"""
words_dico = defaultdict(int)
WP_dico = defaultdict(int)

for doc in nc_zh_treebank:
    for tree in doc:
        len_s = len(tree.words)
        for i in range(1,len_s):
            word = tree.words[i]
            #print(word)
            upos = tree.upos_tags[i]
            #print(upos)
            if upos != "PUNCT" and (not bool(re.search('[a-zA-Z]', word))):
                words_dico[word]+=1
                WP_dico[(word,upos)]+=1


print(len(words_dico))
print(len(WP_dico))

words_lst = sorted(words_dico.items(),key=lambad x:x[1],reverse=True)
WP_lst = sorted(WP_dico.items(),key=lambad x:x[1],reverse=True)
print(words_lst[:300])
print(WP_lst[:300])
#print(WP_dico)
#print(words_dico)

