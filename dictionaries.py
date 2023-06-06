import os
import time
import pandas as pd

path = os.getcwd()

class Dictionaries():

    def __init__(self, dictionary):

        path = os.getcwd()

        self.positive = list(pd.read_csv(path + '/'+ dictionary + '/' + 'positive.csv').iloc[:,1])
        self.negative = list(pd.read_csv(path + '/'+ dictionary + '/' + 'negative.csv').iloc[:,1])

        if dictionary=='LM':
            self.uncertainty = list(pd.read_csv(path + '/'+ dictionary + '/' + 'uncertainty.csv').iloc[:,1])
    
if __name__ == "__main__":

    # test 
    print('\n'+time.strftime('%c') + '\n')

    print('\n ... Loading LM dictionary\n')

    LM = Dictionaries('LM')

    print(LM.negative)

    print('Number of negative words:', len(LM.negative))
    print('Number of positive words:', len(LM.positive))
    print('Number of uncertainty words:', len(LM.uncertainty))


    print('\n...Loading HE dictionary\n')

    HE = Dictionaries('HE')


    print('Number of negative words:', len(HE.negative))
    print('Number of positive words:', len(HE.positive))
    
    print('\n...Loading GI dictionary')

    GI = Dictionaries('GI')

    print('Number of negative words:', len(GI.negative))
    print('Number of positive words:', len(GI.positive))
    print('\n'+time.strftime('%c') + '\n')


