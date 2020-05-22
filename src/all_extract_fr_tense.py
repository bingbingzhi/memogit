#-*-coding:utf-8-*-
import pandas as pd
from decimal import *
from collections import defaultdict

class DepGraph:

    ROOT_TOKEN = '<root>'
    
    def __init__(self,sent_id,vForms,deprel,lemma,feats,edges,wordlist=None,upos_tags=None,with_root=False):
         
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
        self.feats_tense    = [DepGraph.ROOT_TOKEN] + feats
        self.deprel = [DepGraph.ROOT_TOKEN] + deprel
        self.lemma = [DepGraph.ROOT_TOKEN] +lemma
        self.verb_form = [DepGraph.ROOT_TOKEN] + vForms
        

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
            edges   = [ ]
            feats_tense   = [ ]
            deprel = [ ]
            lemma = [ ]
            vForms= [ ]
            for dataline in conll:
                if '-' in dataline[0]:
                    continue          #skips compound word annotation
                words.append(dataline[1])
                upostags.append(dataline[3])
                lemma.append(dataline[2])
                deprel.append(dataline[7].strip())
                if dataline[6] != '0': #do not add root immediately
                    edges.append((int(dataline[6]),dataline[7].strip(),int(dataline[0]))) # shift indexes !
                
                if dataline[5] != '_' and 'VerbForm=Fin' in dataline[5]: # éliminer les verbes participe passé
                    feats_list= dataline[5].split('|')
                    feat_tense = [x for x in feats_list if 'Tense' in x]
                    
                    if feat_tense:
                        feat_mood = feats_list[0]  
                        feats_tense.append((feat_mood,feat_tense[0]))
                    else: 
                        feats_tense.append('_')
                else:
                    feats_tense.append('_')
                if dataline[5] != '_' and 'VerbForm' in dataline[5]:
                    feats_list= dataline[5].split('|')
                    vForm = [x for x in feats_list if 'VerbForm' in x]
                    vForms.extend(vForm)
                else: 
                    vForms.append('_')                    

            return DepGraph(sent_id,vForms,deprel,lemma,feats_tense,edges,words,upos_tags=upostags,with_root=True)

        conll = [ ]
        deptrees = [ ]
        line  = istream.readline( ) # a string
        #checks whether the string consists of whitespace or contains #
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
            label,head = revdeps[node] if node in revdeps else ('root', 0)
            if self.feats_tense:
                L[5] = str(self.feats_tense[node])
            if self.verb_form:
                L[4] = self.verb_form[node]               
            if self.deprel:
                L[7] = self.deprel[node]
            if self.lemma:
                L[2] = self.lemma[node]
            L[6] = str(head)
            lines.append( '\t'.join(L)) 
        return '\n'.join(lines)

    def __len__(self):
        return len(self.words)

def treebank(filename):
    istream = open(filename)
    tlist = []
    deptrees = DepGraph.read_tree(istream)
    #for t in deptrees:
        #print(t)
    while deptrees :
        tlist.append(deptrees)
        deptrees = DepGraph.read_tree(istream)
        #for t in deptrees:
        #    print(t)
        #break
    istream.close()
    return tlist

nc_treebank = treebank('../data/filtered_fr_sample_nc_v15.conll')

n_doc = len(nc_treebank)
#print(len(nc_treebank[0]))

"""
for trees in nc_treebank[:3]:
    for tree in trees: 
        #print('#tree: \n',tree)
        #print('#tree.words: ',tree.words)
        print('sent_id: ',tree.sentence_id)
    print()

        print('#tree.lemma: ',tree.lemma)
        print('#tree.vForms: ',tree.verb_form)        
        print('#tree.gov2dep: ',tree.gov2dep)
        print('#tree.feats: ',tree.feats_tense)
        print('#tree.root_id: ',tree.root_id)
        print('#tree.deprel: ',tree.deprel)
        print()
"""


def root_tense(deptree):
    tense_sent = [ ]
    tense_list = list(deptree.feats_tense) 
    while '_' in tense_list:
        tense_list.remove('_')
    tense_list.remove('<root>')
    if not tense_list:          
        tense_sent = ['UNK']
    # the major tense of a sentence is the root verb tense
    else:
        idx = deptree.root_id
        # if the root has tense attribute, then it's a simple tense
        if deptree.feats_tense[idx]!='_':
            #print(deptree)
            mood = deptree.feats_tense[idx][0].split('=')[1]
            tense = deptree.feats_tense[idx][1].split('=')[1]
            root_dep = deptree.gov2dep[idx]
            edge_verbInf = [x for x in root_dep if deptree.verb_form[x[2]]=="VerbForm=Inf"]
            if tense in ['Imp','Past']:
                tense_sent.append('Past')
            elif tense == 'Pres':
                tense_candidat = 'Pres'
                # capturer le futur proche avec aller(présent)+ verbe infinitive
                if mood == 'Ind' and deptree.lemma[idx]=="aller" and edge_verbInf:
                    tense_candidat = 'Fut'
                elif mood == 'Ind' and deptree.lemma[idx]=="venir" and edge_verbInf:
                    if edge_verbInf[0][2] in deptree.gov2dep:
                        for x in deptree.gov2dep[edge_verbInf[0][2]]:
                            if deptree.lemma[x[2]]=="de":
                                tense_candidat = 'Past'
                elif mood == 'Cnd':
                    tense_candidat='Cnd'
                tense_sent.append(tense_candidat)
            else:
            	tense_sent.append(tense)
            
        # when root doesn't have tense attribute 
        else:
            root_dep = deptree.gov2dep[idx]
            edge_verbInf = [x for x in root_dep if deptree.verb_form[x[2]]=="VerbForm=Inf"] 
            if root_dep:
                # all the children of root who has tense attribute
                edge = [ x for x in root_dep if deptree.feats_tense[x[2]] != '_']
                # the four significant dependency labels
                if len(edge) > 1:
                    edge = [x for x in edge if x[1] in ['aux:pass','aux:tense','cop','aux:caus']]
                if edge:
                    #if len(edge)>1:
                    #    print(edge)
                    #    print(deptree) # no edge has 2 elements 
                    mood = deptree.feats_tense[edge[0][2]][0].split('=')[1]
                    tense = deptree.feats_tense[edge[0][2]][1].split('=')[1]
                    if edge[0][1] in ['aux:pass','cop','aux:caus']:
                        if tense in ['Imp','Past']:
                            tense_sent.append('Past')
                        elif tense == 'Pres':
                            tense_candidat = 'Pres'
                            # capturer le futur proche avec aller(présent)+ verbe infinitive
                            if mood == 'Ind' and deptree.lemma[idx]=="aller" and edge_verbInf:
                                tense_candidat = 'Fut'
                            elif mood == 'Ind' and deptree.lemma[idx]=="venir" and edge_verbInf:
                                if edge_verbInf[0][2] in deptree.gov2dep:
                                    for x in deptree.gov2dep[edge_verbInf[0][2]]:
                                        if deptree.lemma[x[2]]=="de":
                                            tense_candidat = 'Past'
                            elif mood == 'Cnd':
                                tense_candidat='Cnd'
                            tense_sent.append(tense_candidat)
                        else:
                            tense_sent.append(tense)              
                        
                    elif edge[0][1] == 'aux:tense':
                        if tense == 'Pres':
                            if mood == 'Ind':
                                tense_sent.append("PC")
                            elif mood == 'Cnd':
                                tense_sent.append("Cnd")
                            # root error, no sub root 
                            elif mood == 'Sub':
                                te = normalise_tense(tense_list,deptree)
                                print(te,'\n',deptree)
                                tense_sent.append(te)
                        elif tense == 'Fut':
                            tense_sent.append("Fut")
                        #  plus-que-pafait; passé antérieur; futur antérieur; 
                        #    # les temps rares
                        # all the complexe tense should be put into category "Past"
                        else:
                            tense_sent.append('Past')
                        
                    else:
                        #- edge == 'conj' owing to root annotated errors 
                        # - edge == parataxis, no root verb
                        # - ddge == acl:relcl, no root verb
                        tense_sent.append(normalise_tense(tense_list,deptree))
                        #print(deptree)
                        #print(normalise_tense(tense_list,deptree))
                    
                
                else:
                    #- root children dont have tense attribute
                    #  root grandson has tense attribute (2simple),
                    #     1 complexe due to annotated error, the generated tense is present, it turns out to be correct.
                    tense_sent.append(normalise_tense(tense_list,deptree))
                    #print(deptree)
                    #print(normalise_tense(tense_list,deptree))
                    
            else:
                print(deptree)
                print("Error: root dosen't have children nodes")

    return tense_sent

# for sentences that we can't get root tense infos neither from root nor from root deps
def normalise_tense(tense_list,deptree):
    
    tense_idx = [i for i in range(1,len(deptree.feats_tense)) if deptree.feats_tense[i] != '_' ]
    if not tense_idx:
        gold_tense = "UNK"
    else:
        tenses = [ ]
        for idx in tense_idx:
                   # print(tense_idx)
                   # print(deptree.feats_tense)
                    mood = deptree.feats_tense[idx][0].split('=')[1]
                    t = deptree.feats_tense[idx][1].split('=')[1]
                    if deptree.upos_tags[idx] == "AUX" and deptree.deprel[idx]=="aux:tense":
                            if t == 'Pres'and mood=='Ind':
                                tense="PC"
                            elif t == 'Pres' and mood=='Cnd':
                                tense='Cnd'
                            elif t == 'Pres' and mood == 'Sub':
                                tense= 'UNK'
                            elif t == 'Fut':
                                tense="Fut"
                            #  plus-que-pafait; passé antérieur; futur antérieur; 
                            # cond passé; impératif passé; sub.passé ou sub.plus-que-parfait  # les temps rares
                            # all the complexe tense should be put into category "Past"
                            else:
                                tense='Past'				
                    elif idx not in deptree.gov2dep:
                        if t in ['Imp','Past']:
                            tense = 'Past'
                        elif t == 'Pres' and mood == "Cnd":
                            tense = 'Cnd'
                        else:
                            tense = t
                    else:
                            root_dep = deptree.gov2dep[idx]
                            edge_verbInf = [x for x in root_dep if deptree.verb_form[x[2]]=="VerbForm=Inf"]                        
                            if t in ['Imp','Past']:
                                tense='Past'
                            elif t == 'Pres':
                                # capturer le futur proche avec aller(présent)+ verbe infinitive
                                if mood == 'Ind' and deptree.lemma[idx]=="aller" and edge_verbInf:
                                    tense='Fut'
                                elif mood == 'Ind' and deptree.lemma[idx]=="venir" and edge_verbInf:
                                    if edge_verbInf[0][2] in deptree.gov2dep:
                                        for x in deptree.gov2dep[edge_verbInf[0][2]]:
                                            if deptree.lemma[x[2]]=="de":
                                                tense='Past'
                                                #print("#normalise tense: \n",deptree)
                                            else:
                                                tense='Pres'
                                    else:
                                        tense='Pres'
                                elif mood == 'Cnd':
                                    tense='Cnd'
                                else:
                                    tense = t
                            else:
                                tense = t
                    #tense= deptree.feats_tense[idx].split('=')[1]
                    tenses.append(tense)
        if len(set(tenses)) == 1:
                gold_tense = tenses[0]
        else:
                gold_tense = "UNK"
                #print("tense-conflict: \n",deptree)
    return gold_tense


sent2tense = defaultdict(str)
for doc in nc_treebank:
    for tree in doc:
        tense = root_tense(tree)
        if len(tense)!=1:
            print("error! more thant one major tense:\n ", tree.sentence_id,tense,'\n',tree)
        sent2tense[tree.sentence_id]= tense[0]
    
    if len(doc)>99:
        print("error! doc length > 99")

print("length of tenses dict: ",len(sent2tense))
result = pd.value_counts(list(sent2tense.values()))
print(result)
#print(sent2tense)
#print(n_doc)
#print(count)

