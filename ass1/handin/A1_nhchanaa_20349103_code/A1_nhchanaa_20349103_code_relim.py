# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 22:12:48 2016

@author: dsm
"""

from pymining import itemmining, assocrules
import time


class freq_mining(object):
	"""docstring for ClassName"""
	def __init__(self, transactions, min_sup, min_conf):
		
		self.transactions = transactions  # database
		self.min_sup = min_sup  # minimum support
		self.min_conf = min_conf  # minimum support

	def freq_items(self):

		relim_input = itemmining.get_relim_input(self.transactions)
		item_sets = itemmining.relim(relim_input, self.min_sup)
		return item_sets

	def association_rules(self):

		item_sets = self.freq_items()
		rules = assocrules.mine_assoc_rules(item_sets, self.min_sup, self.min_conf)
		return rules

def main(transactions, min_sup, min_conf):

	item_mining = freq_mining(transactions, min_sup, min_conf)
	freq_items = item_mining.freq_items()
	#rules = item_mining.association_rules()

	#print(freq_items)
	#print rules
	
	''' Write frequent item set to a file'''
	file = open('frequentItemSet_relim.txt', 'w')

	for key, val in freq_items.items():
		for item in key:
			file.write("%s " % item)
			#print(item,end=' ')
		#file.write("%d\n" % val)
		file.write("\n")
		#print()
	file.close()


if __name__ == "__main__":
        start_time = time.time()

        '''
        transactions = (('a','b','c','d','e','f','g','h'),
                    ('a','f','g'),
                    ('b','d','e','f','j'),
                    ('a','b','d','i','k'),
                    ('a','b','e','g'))
        '''
       
        #min_sup = 10
        min_sup = 100
        min_conf = 0

        with open('freq_items_dataset.txt') as file:
        #with open('tmp.txt') as file:
                array =  [[int(digit) for digit in line.split()] for line in file]
        #print(array)
        file.close()

        main(array, min_sup, min_conf)
        print("--- %s seconds ---" % (time.time() - start_time))









