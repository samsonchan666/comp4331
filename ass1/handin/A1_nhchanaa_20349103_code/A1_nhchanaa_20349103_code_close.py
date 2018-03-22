import time

def main(itemsets):
    
    closedItemsets = {}
    maxItemsets = {}
    ''' Write close itemset and max frequent itemset to a file'''
    file = open('closedItemSet.txt', 'w')
    _file = open('maxItemSet.txt', 'w')

    '''2-level for loop to check itemset's immediate superset '''
    for key, val in itemsets.items():
        closed = True
        max = True
        for _key, _val in itemsets.items():
            '''Check if it is immediate superset'''
            if len(key)+1 == len(_key) and key != _key:
                if  _key.issuperset(key) :
                    """Not a max frequent itemset if it has a superset"""
                    max = False
                    if _val == val:
                        '''Not close if a immediate superset 
                        has the same frequency as the itemset'''
                        closed = False
                        break
        '''Record the result in dictionary'''
        if (closed):
            closedItemsets[key] = val
        if (max):
            maxItemsets[key] = val

    #print(closedItemsets)
    """Save the result to file"""
    for key in closedItemsets:
        for item in key:
            file.write("%s " % item)
        file.write("\n")
    file.close()

    for key in maxItemsets:
        for item in key:
            _file.write("%s " % item)
        _file.write("\n")
    _file.close()

if __name__ == "__main__":
        start_time = time.time()
        itemsets = {}
        '''Load frequent item sets which contain the frequency in last item in each transaction'''
        with open('frequentItemSet_withfreq.txt') as file:
            for line in file:
                _line = line.split();
                freq = _line.pop()
                itemsets[frozenset(_line)] = freq;
        file.close()

        main(itemsets)
        print("--- %s seconds ---" % (time.time() - start_time))