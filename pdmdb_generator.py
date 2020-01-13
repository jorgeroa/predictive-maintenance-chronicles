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

from seq_generation.chronicle_generator import *


class disturbed_chro_sequence(chro_sequence):
    """ A sequence is a list of time-stamped items
    
    All sequences start at 0 and there duration is defined while constructed (self.duration)
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


class disturbed_chrosequence_generator(sequence_generator):
    """Factory for sequence based on disturbed chronicles
    """
    def __init__(self):
        super().__init__()
    
    def generate(self,l):
        return disturbed_chro_sequence(l)

class disturbed_chronicle_generator(chronicle_generator):
    """Factory class for disturbed chronicles
    
    It provides the function generate_similar to generate disturbed chronicles
    """
    
    def __init__(self, ig, cg, cd=0.3):
        super().__init__(ig, cg, cd)
        
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


class pdmdb_generator(db_generator):
    """ PdM generator: Generator of a dataset containing sequences for predictive maintenance (PdM).
        The generator can generate both disturbed and non-disturbed event data from a set of chronicles.
    
    """

    def __init__(self, nbitems=100, l = 30, lp=4, fl="uniform", dc= 0.30, minstart=-100, maxstart=100, minduration=0.1, maxduration=200):
        """Constructor of the db generator
        
        :param nbitems: vocabulary size (default 100)
        :param l: mean length of the sequences
        :param fl: item frequency distribution 'uniform', 'gaussian'
        :param dc: constraint density (if 0: no constraints, if 1 each pair of event are temporaly constraints)
        :param lp: pattern length
        :param minstart, maxstart, minduration, maxduration: temporal constraint characteristics
        """
        
        itemgen= item_generator(n=nbitems, fl=fl)
        seqgen = disturbed_chrosequence_generator() # Create instance of the generator of disturbed sequences
        constrgen = constraint_generator(minstart, maxstart, minduration, maxduration)
        patgen = disturbed_chronicle_generator(itemgen, constrgen, dc)
        pattern_generator.lpat=lp #change class attribute  
    
        super().__init__(itemgen, seqgen, patgen)

    
    def generate(self, nb=None, l=None, npat=None, th=None, patterns=None):
        """Generation of the sequence database
        
        :param nb: number of sequences
        :param l: mean length of the sequence
        :param npat: number of patterns
        :param th: frequency threshold (number of sequences covering each pattern)
        :param pattern: a list of chronicles
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
            
        # JGT: Change from original code to support generation of sequences from existing pattern (chronicles)
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
            self.db[i].self_generate(self.item_generator)
        
        return self.db

if __name__ == "__main__":

    # nbitems: vocabulary size // l: mean length of the sequences // lp: pattern length
    generator=pdmdb_generator(nbitems=20,l=8, lp=4, dc=0, minstart=0.1, minduration=1, maxduration=5)
    
    # sequences: stores all the chronicle sequences generated by the generator 
    # nb: 10 sequences // l: 7 (mean length of the sequence) // npat: 3 patterns // th: .5 (frequency threshold (number of sequences covering each pattern))
    sequences = generator.generate(nb=10, l=7, npat=3, th=.5)

    # The disturbed chronicles to be used for generating disturbed sequences
    disturbed_chronicles = []

    print("======== PATTERNS =======")
    print(generator.output_patterns())

    print("======== SEQUENCES =======")
    for s in sequences:
        print(str(s))

    print("================================")

    for c in generator.patterns:
        # Probability of:
        # 1- removing items
        # 2- modifying constraints
        # 3- adding item
        disturbed_chronicle = generator.pattern_generator.generate_similar(c,[0,1,0])
        # print(str(disturbed_chronicle))
        disturbed_chronicles.append(disturbed_chronicle)

    # sequences: stores all the chronicle sequences generated by the generator 
    # nb: 10 sequences // l: 7 (mean length of the sequence) // npat: 3 patterns // th: .5 (frequency threshold (number of sequences covering each pattern))
    # patterns=chronicles: provides the disturbed chronicles to the generator to generate new sequences
    disturbed_sequences = generator.generate(nb=10, l=7, npat=3, th=.5, patterns=disturbed_chronicles)

    print("======== DISTURBED PATTERNS =======")
    print(generator.output_patterns())

    print("======== DISTURBED SEQUENCES =======")
    for s in disturbed_sequences:
        print("Absolute: ",str(s))
        print("Relative: ",str(s.write_line_relative()))

