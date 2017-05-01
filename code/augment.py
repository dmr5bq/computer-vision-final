from AugmentorOptimized import *

__author__ = "Dominic Ritchey"

aug = AugmentorOptimized()

aug.load_from('person4/seq1')
aug.run()
aug.load_from('person4/seq1/output')
aug.run()

aug.load_from('person4/seq2')
aug.run()
aug.load_from('person4/seq2/output')
aug.run()

aug.load_from('person4/seq3')
aug.run()
aug.load_from('person4/seq3/output')
aug.run()