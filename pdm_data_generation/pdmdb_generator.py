#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Module for random generation of event data for useful for predictive maintenance (PdM) analysis.

.. ::see db_generator
"""

__author__ = "Jorge Roa"
__email__ = "jorge.roa@insa-strasbourg.fr"
__email__ = "jroa@frsf.utn.edu.ar"

import numpy as np
import sys, getopt
import warnings
import math

from random import randint

from seq_generation.chronicle_generator import *


class disturbed_chro_sequence(chro_sequence):
    """ A sequence is a list of time-stamped items
    
    All sequences start at 0 and the duration is defined while constructed (self.duration)
    """
    gen_int_timestamped=True #if True, timestamped will be integers

    # ################################################
    # IT IS NECESSARY TO ADD HERE THE CODE FOR GENERATING THE DISTURBED DATA. SEE THE CODE FROM SUPERCLASS chro_sequence.
    # ################################################
    
    def __init__(self, rl =20, d=1000):
        super().__init__(rl,d)

    def write_line_relative(self):
        """Write out function: 1 sequence in a line with relative times instead of absolute
        """
        if len(self.seq)==0:
            return ""
        s=" (0,"+str(self.seq[0][1])+")"
        d=0
        t0=self.seq[0][0]
        for c in self.seq[1:]:
            d=c[0]-t0
            s += " ("+str(d)+","+str(c[1])+")"
            t0=c[0]
        return s

    def disturb(self,interval,s):
        test=max(interval[0]-s,0)==0
        m=min(interval[1],self.duration)
        if(np.random.random()> .5 and  not test ):
            t = np.random.uniform(interval[0]-s,interval[0]-1)
        else:
            t = np.random.uniform(m+1,m+s)
        return t
    
    # def self_generate(self, item_gen,pert=-1):
    #     """ Generate the sequence of items from the patterns it contains and according to the required length
    
    #     Timestamped are generated according to a random uniform law within the duration of the sequence
    #     The function enable to generate negative pattern
    
    #     :param item_gen: random item generator to mix the pattern occurrences with random items.
    #     """
    
    #     # no patterns inside: fully random sequences
    #     if len(self.patterns)==0:
    #         l=int(np.random.normal(self.requiredlen, float(self.requiredlen)/float(10.0)))
    #         for i in range(l):
    #             #random item (according to an item generator)
    #             #item = item_gen.generate()
    #             item=-1
    #             #random timestamp
    #             timestamp = np.random.uniform(self.duration)
    #             self.seq.append( (timestamp,item) )
    #             if chro_sequence.gen_int_timestamped:
    #                 self.seq = [ (int(c[0]),c[1]) for c in self.seq ]
    #             self.seq.sort(key=lambda tup: tup[0])
    #         return 
            
    #     negative_period={}
        
    #     totcreated=0
        
    #     #for p in self.patterns:
    #     i=randint(0, len(self.patterns)-1)
    #     p=self.patterns[i]
    #     occurrence=[] #timestamped of the occurrence
    #     t=int(np.random.uniform(0,self.duration/2)) #chronicle start at the beginning of the sequence
    #     occurrence.append(t)
    #     self.seq.append( (t,p[0]) )
    #     npert=0

    #     for i in range(1,len(p)):
    #         # construct a interval in which i-th event could occurs
    #         # interval=[0,100000]
    #         interval=[0,self.duration]
    #         last_e=-1

    #         lc = p[ (i-1,i) ]
    #         interval[0]=occurrence[i-1]+lc[0]
    #         last_e=i-1
    #         interval[1]=occurrence[i-1]+lc[1]

    #         t = np.random.uniform(interval[0],interval[1])
    #         self.seq.append( (int(t),p[i]) )
    #         occurrence.append(int(t)) #timestamp of the i-th item in the chronicle occurrence
            
    #     if not p.negative_position is None:
    #         if p.negative_position == (len(occurrence)-1):
    #             negative_period[p]=(occurrence[p.negative_position],float("inf"))
    #         else:
    #             negative_period[p]=(occurrence[p.negative_position],occurrence[p.negative_position+1])
            
    #     totcreated += len(p)
            
    #     # l=int(np.random.normal(self.requiredlen, float(self.requiredlen)/float(10.0)))
    #     # while totcreated<l:
    #     #     #random item (according to an item generator)
    #     #     #item = item_gen.generate()
    #     #     item=-1
    #     #     #random timestamp
    #     #     timestamp = np.random.uniform(self.duration)
    #     #     self.seq.append( (timestamp,item) )
    #     #     totcreated+=1
            
            
    #     #sort the list according to timestamps
    #     if chro_sequence.gen_int_timestamped:
    #         self.seq = [ (int(c[0]),c[1]) for c in self.seq ]
    #     self.seq.sort(key=lambda tup: tup[0])
        
        
    #     for p in self.patterns:
    #         if not p.negative_position is None:
    #             for item in self.seq:
    #                 if item[0]>negative_period[p][0] and item[0]<negative_period[p][1]:
    #                     p.add_possible_neg_item(item[1], self.id)

    # def self_generate(self, item_gen):
    #     """ Generate the sequence of items from the patterns it contains and according to the required length
    
    #     Timestamped are generated according to a random uniform law within the duration of the sequence
    #     The function enable to generate negative pattern
    
    #     :param item_gen: random item generator to mix the pattern occurrences with random items.
    #     """
    
    #     # # no patterns inside: fully random sequences
    #     # if len(self.patterns)==0:
    #     #     l=int(np.random.normal(self.requiredlen, float(self.requiredlen)/float(10.0)))
    #     #     for i in range(l):
    #     #         #random item (according to an item generator)
    #     #         item = item_gen.generate()
    #     #         #random timestamp
    #     #         timestamp = np.random.uniform(self.duration)
    #     #         self.seq.append( (timestamp,item) )
    #     #         if chro_sequence.gen_int_timestamped:
    #     #             self.seq = [ (int(c[0]),c[1]) for c in self.seq ]
    #     #         self.seq.sort(key=lambda tup: tup[0])
    #     #     return
            
    #     negative_period={}
        
    #     totcreated=0
    #     for p in self.patterns:
    #         occurrence=[] #timestamped of the occurrence
    #         t=np.random.uniform(0,self.duration/2) #chronicle start at the beginning of the sequence
    #         occurrence.append(t)
    #         self.seq.append( (t,p[0]) )
            
    #         for i in range(1,len(p)):
    #             # construct a interval in which i-th event could occurs
    #             interval=[0,self.duration]
    #             for j in range(i):
    #                 lc = p[ (j,i) ] #chronicle constraint between j and i
    #                 #interval intersection (constraints conjunction)
    #                 if interval[0]<occurrence[j]+lc[0]:
    #                     interval[0]=occurrence[j]+lc[0]
    #                 if interval[1]>occurrence[j]+lc[1]:
    #                     interval[1]=occurrence[j]+lc[1]
                    
    #             #generate a timestamp in the interval
    #             if interval[0]>=interval[1]:
    #                 warnings.warn("*** chronicle is not consistent ***")
    #                 self.seq=[]
    #                 return
                
    #             t = np.random.uniform(interval[0],interval[1])
    #             self.seq.append( (t,p[i]) )
    #             occurrence.append(t) #timestamp of the i-th item in the chronicle occurrence
                
    #         # if not p.negative_position is None:
    #         #     if p.negative_position == (len(occurrence)-1):
    #         #         negative_period[p]=(occurrence[p.negative_position],float("inf"))
    #         #     else:
    #         #         negative_period[p]=(occurrence[p.negative_position],occurrence[p.negative_position+1])
                
    #         totcreated += len(p)
            
    #     # l=int(np.random.normal(self.requiredlen, float(self.requiredlen)/float(10.0)))
    #     # while totcreated<l:
    #     #     #random item (according to an item generator)
    #     #     item = item_gen.generate()
    #     #     #random timestamp
    #     #     timestamp = np.random.uniform(self.duration)
    #     #     self.seq.append( (timestamp,item) )
    #     #     totcreated+=1
            
            
    #     #sort the list according to timestamps
    #     if chro_sequence.gen_int_timestamped:
    #         self.seq = [ (int(c[0]),c[1]) for c in self.seq ]
    #     self.seq.sort(key=lambda tup: tup[0])
        
        
    #     # for p in self.patterns:
    #     #     if not p.negative_position is None:
    #     #         for item in self.seq:
    #     #             if item[0]>negative_period[p][0] and item[0]<negative_period[p][1]:
    #     #                 p.add_possible_neg_item(item[1], self.id)

    def self_generate(self, item_gen,pert=-1):
        """ Generate the sequence of items from the patterns it contains and according to the required length
    
        Timestamped are generated according to a random uniform law within the duration of the sequence
        The function enable to generate negative pattern
    
        :param item_gen: random item generator to mix the pattern occurrences with random items.
        """
    
        # no patterns inside: fully random sequences
        if len(self.patterns)==0:
            l=int(np.random.normal(self.requiredlen, float(self.requiredlen)/float(10.0)))
            for i in range(l):
                #random item (according to an item generator)
                #item = item_gen.generate()
                item=-1
                #random timestamp
                timestamp = np.random.uniform(self.duration)
                self.seq.append( (timestamp,item) )
                if chro_sequence.gen_int_timestamped:
                    self.seq = [ (int(c[0]),c[1]) for c in self.seq ]
                self.seq.sort(key=lambda tup: tup[0])
            return 
            
        negative_period={}
        
        totcreated=0
        
        #for p in self.patterns:
        i=randint(0, len(self.patterns)-1)
        p=self.patterns[i]
        occurrence=[] #timestamped of the occurrence
        t=int(np.random.uniform(0,self.duration/2)) #chronicle start at the beginning of the sequence
        occurrence.append(t)
        self.seq.append( (t,p[0]) )
        npert=0

        for i in range(1,len(p)):
            # construct a interval in which i-th event could occurs
            interval=[0,100000]
            last_e=-1
            """
            for j in range(i):
                lc = p[ (j,i) ] #chronicle constraint between j and i
                #interval intersection (constraints conjunction)
                if interval[0]<occurrence[j]+lc[0]:
                    interval[0]=occurrence[j]+lc[0]
                    last_e=j
                if interval[1]>occurrence[j]+lc[1]:
                    interval[1]=occurrence[j]+lc[1]
            if(pert>=0 and len(p)>7):
                lc = p[ (i-1,i) ]
                interval[0]=occurrence[i-1]+lc[0]
                last_e=i-1
                interval[1]=occurrence[i-1]+lc[1]   
            #generate a timestamp in the interval
            if interval[0]>=interval[1]:
                warnings.warn("*** chronicle is not consistent ***")
                self.seq=[]
                return
            """
            lc = p[ (i-1,i) ]
            interval[0]=occurrence[i-1]+lc[0]
            last_e=i-1
            interval[1]=occurrence[i-1]+lc[1]
            #pert=1 ==>pert fonctionnement abnormal
            #pert=0 ==>bruit
            #pert=-1 ==>pas pert 
            #pert=-2 LSTM train
            if(pert>=0 and last_e!=-1 and i>2):

                if(npert==0):
                    if(pert==1):
                        inter=[interval[0]-20,interval[1]+20]
                        t=self.disturb(inter,100)
                    elif(pert==0):
                        d=interval[0] - occurrence[i-1]
                        if(d<0):
                            t=self.disturb(interval,10)
                        else:
                            s=min(10,d)
                            t=self.disturb(interval,s)
                    if interval!=[0,self.duration]:
                        npert +=1
                elif(np.random.random()> .5 and pert==1 ):
                    #t=self.disturb(interval,10)
                    inter=[interval[0]-20,interval[1]+20]
                    t=self.disturb(inter,100)
                else:
                    t = np.random.uniform(max(interval[0],0),interval[1])
            elif (pert==-2):
                #t = np.random.uniform(interval[0],interval[1])
                t =(interval[1]+interval[0])/2
            else:
                t = np.random.uniform(interval[0],interval[1])
            self.seq.append( (int(t),p[i]) )
            occurrence.append(int(t)) #timestamp of the i-th item in the chronicle occurrence
            
        if not p.negative_position is None:
            if p.negative_position == (len(occurrence)-1):
                negative_period[p]=(occurrence[p.negative_position],float("inf"))
            else:
                negative_period[p]=(occurrence[p.negative_position],occurrence[p.negative_position+1])
            
        totcreated += len(p)
            
        l=int(np.random.normal(self.requiredlen, float(self.requiredlen)/float(10.0)))
        while totcreated<l:
            #random item (according to an item generator)
            #item = item_gen.generate()
            item=-1
            #random timestamp
            timestamp = np.random.uniform(self.duration)
            self.seq.append( (timestamp,item) )
            totcreated+=1
            
            
        #sort the list according to timestamps
        if chro_sequence.gen_int_timestamped:
            self.seq = [ (int(c[0]),c[1]) for c in self.seq ]
        self.seq.sort(key=lambda tup: tup[0])
        
        
        for p in self.patterns:
            if not p.negative_position is None:
                for item in self.seq:
                    if item[0]>negative_period[p][0] and item[0]<negative_period[p][1]:
                        p.add_possible_neg_item(item[1], self.id)


class disturbed_chrosequence_generator(sequence_generator):
    """Factory for sequence based on disturbed chronicles
    """
    def __init__(self,maxduration=1000):
        super().__init__()
        self.maxduration = maxduration
    
    def generate(self,l):
        return disturbed_chro_sequence(rl=l,d=self.maxduration)


class disturbed_chronicle_generator(chronicle_generator):
    """Factory class for disturbed chronicles
    
    It provides the function generate_similar to generate disturbed chronicles
    """
    
    def __init__(self, ig, cg, cd=0.3):
        super().__init__(ig, cg, cd)

    # CHECK IF IT IS REALLY NECESSARY TO REIMPLEMENT THIS METHOD 
    def __raw_generate__(self, l):
        chro = chronicle()
        for i in range(l):
            item = self.itemGenerator.generate()
            chro.add_item(item)
        if(l<7):
            for i in range(l):
                for j in range(i+1,l):
                    if np.random.rand()<self.constraintdensity:
                        c=self.constraintGenerator.generate("after")
                        chro.add_constraint(i, j, c)
        else:
            for i in range(l):
                #if np.random.rand()<self.constraintdensity:
                if np.random.rand()<1:
                    c=self.constraintGenerator.generate("after")
                    chro.add_constraint(i, i+1, c)

        return chro
        
    def generate_similar(self, C, proba=[0.1,0.8,0.1]):
        """function that generates a chronicle similar to C
        
        representing the proba of modifications
            1- removing items (defailt 0.1)
            2- modyfying constraints (default 0.8)
            3- adding item (default 0.1)
        
        what can change in the generated chronicle:
            - temporal bounds
            - multi-set (add or remove items)
        
        :param C: is a chronicle
        :param proba: list of 3 values representing the proba of modifications
        :return: a chronicle
        """
        if not isinstance(C, chronicle):
            return
            
        chro = chronicle()
        
        ########## RANDOM MODIFICATION SELECTION  ###############
        removeitem=False
        modify_tconst = False
        additem=False
        
        #proba normalisation
        vec=np.array(proba)
        proba = vec/np.sum(vec)
        alea=np.random.rand()
        i=0
        while i<3 and alea>proba[i]:
            alea -= proba[i]
            i+=1
        if i==0:
            removeitem=True
        elif i==1:
            modify_tconst = True
        else:
            additem=True
        
        ################ CHRONICLE MODIFICATIONS #############
        l=len(C.sequence)
        if removeitem:
            idr = np.random.randint( l )
            for i in range(l):
                if i==idr:
                    continue
                chro.sequence.append( C[i] )
            #copy constraints (removing idr + decay references)
            for i in range(idr):
                for j in range(i+1,idr):
                    chro.add_constraint(i, j, C[(i,j)] )
                for j in range(idr+1,l):
                    chro.add_constraint(i, j-1, C[(i,j)] )
            for i in range(idr+1,l):
                for j in range(i+1,l):
                    chro.add_constraint(i-1, j-1, C[(i,j)] )
                    
        if additem: #add a new item to
            chro.sequence = list(C.sequence)
            chro.tconst = C.tconst.copy()
            ni = self.itemGenerator.generate()
            chro.sequence.append(ni)
            nl = len(chro.sequence)-1
            for j in range( nl ):
                if np.random.uniform(0,1)<self.constraintdensity:
                    c=self.constraintGenerator.generate()
                else:
                    c=(-float("inf"), float("inf"))
                chro.add_constraint(j, nl, c)
        
        if modify_tconst:
            chro.sequence = list(C.sequence)
            chro.tconst = dict(C.tconst)
            j = np.random.randint( 1, l )
            i = np.random.randint( j )
            
            #generate a new random constraint
            c = self.constraintGenerator.generate()
            chro.add_constraint(i, j, c )
        
        ################ CHRONICLE MINIMISATION #############
        chro.minimize()
        return chro

class disturbed_constraint_generator(constraint_generator):
    """Chronicle constraint generator
    
    It randomly generates temporals constraints for chronicles, ie temporal intervals.
    The interval boundaries are uniformly distributed within the limits. 
    """
    def __init__(self, minstart=-100, maxstart=100, minduration=0.1, maxduration=200):
        super().__init__(minstart, maxstart, minduration, maxduration)
    
    def generate(self, ctype=""):
        if ctype=="after":
            s= np.random.uniform(0, self.Ms)
            f= s + np.random.uniform(self.md, self.Md)
        else:
            s= np.random.uniform(self.ms, self.Ms)
            f= s + np.random.uniform(self.md, self.Md)
        c=(int(s), int(f))  # JGT: Changed from superclass method
        return c

class pdmdb_generator(db_generator):
    """ PdM generator: Generator of a dataset containing sequences for predictive maintenance (PdM).
        The generator can generate both disturbed and non-disturbed event data from a set of chronicles.
    
    """

    def __init__(self, nbitems=100, lp=4, fl="uniform", dc= 0.30, minstart=-100, maxstart=100, minduration=0.1, maxduration=200):
        """Constructor of the db generator
        
        :param nbitems: vocabulary size (default 100)
        :param l: mean length of the sequences
        :param fl: item frequency distribution 'uniform', 'gaussian'
        :param dc: constraint density (if 0: no constraints, if 1 each pair of event are temporaly constraints)
        :param lp: pattern length
        :param minstart, maxstart, minduration, maxduration: temporal constraint characteristics
        """
        
        itemgen= item_generator(n=nbitems, fl=fl)
        seqgen = disturbed_chrosequence_generator(maxduration) # Create instance of the generator of disturbed sequences
        constrgen = disturbed_constraint_generator(minstart, maxstart, minduration, maxduration)
        patgen = disturbed_chronicle_generator(itemgen, constrgen, dc)
        pattern_generator.lpat=lp #change class attribute 

        super().__init__(itemgen, seqgen, patgen)


    # def generate_sequences(self):
    #     self.db=[]
    #     sequence.nbs=0
    #     for i in range(self.nbex):
    #         self.db.append( self.sequence_generator.generate(self.l,self.maxduration) )


    def generate(self, nb=None, l=None, npat=None, th=None, patterns=None, pert=-1):
        """Generation of the sequence database
        
        :param nb: number of sequences
        :param l: mean length of the sequence
        :param npat: number of patterns
        :param th: frequency threshold (number of sequences covering each pattern)
        :param pattern: a list of chronicles. Used to generate a set of disturbed sequences from normal chronicles
        """
        #update the parameters of the generation
        if not nb is None:
            self.nbex=nb
        if not l is None:
            self.l=l
        if not th is None:
            self.th=th
        if not npat is None:
            self.nbpat=npat
            
        #collect "real" statistics about the generated sequences
        self.stats={}
        
        #generate a set of nbex empty sequences
        self.generate_sequences()
        self.stats["nbex"]=self.nbex
            
        # JGT: Change from original code to support generation of disturbed sequences from existing pattern (chronicles)
        if(patterns==None):
            #generate self.nbpat random patterns using the generator
            self.patterns=self.generate_patterns()
        else:
            self.patterns=patterns
        self.stats["nbpat"] = len(self.patterns)
        
        nbM=-1
        nbMean=0
        nbm=self.nbex
        #attribute transactions to the patterns
        for p in self.patterns:
            nbocc = self.nbex*self.th  + (np.random.geometric(0.15)-1) # number of occurrences generated (randomly above the threshold)
            nbocc = min(nbocc, self.nbex)
            #generation of a random set of sequences of size nbex (ensure no repetitions)
            vec = list(range(0,self.nbex))    # get a set of id
            np.random.shuffle( vec )    # shuffle it a little bit
            patpos = vec[:int(nbocc)]   # take just take the require number of sequence (at the beginning)
            p.tidl = patpos
            
            nb=0
            for pos in patpos:
                try:
                    self.db[pos].add_pattern( p )
                    nb+=1
                except IndexError:
                    warnings.warn("*** index error: "+str(pos)+" ***")
                    pass
            nbM=max(nbM,nb)
            nbM=min(nbm,nb)
            nbMean+=nb
        if self.stats["nbpat"]!=0: 
            nbMean/=self.stats["nbpat"]
        else:
            nbMean=0
        self.stats["nboccpat_max"]=nbM
        self.stats["nboccpat_min"]=nbm
        self.stats["nboccpat_mean"]=nbMean
        
        #generate sequences
        for i in range(self.nbex):
            self.db[i].self_generate(self.item_generator,pert)
        
        return self.db

if __name__ == "__main__":

    # #########################################
    # Parameters setting for the generator
    n_events = 10   # Number of events
    events_per_pattern = 10  # Number of events per pattern
    constraint_density = 0.1    # Percentaje of connected events that will have constraints (1 means all connections will have constraints, 0 the opposite)
    min_start = 0   # Time of minimun start of an event
    min_duration = 1    
    max_duration = 300    # Max duration for a time constraint (5 hours=60*5=300).
    # #########################################

    
    # #########################################
    # Parameters setting for the generation of sequences using the generator
    n_sequences = 10    # Number of sequences to be generated
    sequences_mean_lenght = 10  # Maximun length of a sequence
    n_patterns = 1  # Number of patterns to generate
    pattern_coverage = 0.3 # Percentaje of sequences covering each pattern
    # #########################################

    # nbitems: number of events // lp: number of events per pattern
    generator=pdmdb_generator(nbitems=n_events, 
                                lp=events_per_pattern, 
                                dc=constraint_density, 
                                minstart=min_start, 
                                minduration=min_duration, 
                                maxduration=max_duration)
    
    # sequences: stores all the chronicle sequences generated by the generator 
    sequences = generator.generate(nb=n_sequences, 
                                    l=sequences_mean_lenght, 
                                    npat=n_patterns, 
                                    th=pattern_coverage)

    # The disturbed chronicles to be used for generating disturbed sequences
    disturbed_chronicles = []

    print("======== PATTERNS =======")
    print(generator.output_patterns())

    print("======== SEQUENCES =======")
    for s in sequences:
        print("Absolute: ",str(s))
        print("Relative: ",str(s.write_line_relative()))

    print("================================")

    for c in generator.patterns:
        # Probability of: [1:removing items, 2:modifying constraints, 3:adding items]
        disturbed_chronicle = generator.pattern_generator.generate_similar(c,[0,1,0])
        # print(str(disturbed_chronicle))
        disturbed_chronicles.append(disturbed_chronicle)

    # sequences: stores all the chronicle sequences generated by the generator 
    disturbed_sequences = generator.generate(nb=n_sequences, 
                                    l=sequences_mean_lenght, 
                                    npat=n_patterns, 
                                    th=pattern_coverage,
                                    patterns=disturbed_chronicles)

    print("======== DISTURBED PATTERNS =======")
    print(generator.output_patterns())

    print("======== DISTURBED SEQUENCES =======")
    for s in disturbed_sequences:
        print("Absolute: ",str(s))
        print("Relative: ",str(s.write_line_relative()))

