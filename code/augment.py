from Augmentor import *

aug = Augmentor()

aug.load_from('person4/seq1')
aug.run()

aug.load_from('person4/seq2')
aug.run()

aug.load_from('person4/seq3')
aug.run()