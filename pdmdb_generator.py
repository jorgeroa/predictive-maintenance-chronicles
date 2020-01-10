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

class disturbed_chrosequence_generator(sequence_generator):
    """Factory for sequence based on disturbed chronicles
    """
    def __init__(self):
        super().__init__()
    
    def generate(self,l):
        return disturbed_chro_sequence(l)


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
        patgen = chronicle_generator(itemgen, constrgen, dc)
        pattern_generator.lpat=lp #change class attribute  
    
        super().__init__(itemgen, seqgen, patgen)


if __name__ == "__main__":

    # nbitems: vocabulary size // l: mean length of the sequences // lp: pattern length
    generator=pdmdb_generator(nbitems=20,l=8, lp=4)
    
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
        print(str(s))

