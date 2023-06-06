# Author: Cássio Roberto de Andrade Alves 
# First version: 29 february 2020

# load packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from nltk.corpus import stopwords
import time

# import nltk
# nltk.download('stopwords')

from copom_statements.dictionaries import *

# load dictionary. Options: 'LM', 'HE' and 'GI'
dict_name = ['LM'] # ['LM', 'HE', 'GI']

for name in dict_name:

    dic = Dictionaries(name) # instanciate the dictionary

    negative_words = dic.negative # load negative words
    positive_words = dic.positive # load positive words

    if name=='LM':
        uncertainty_words = dic.uncertainty # if 'LM' was choosen, then load uncertainty word

    # print some info
    if name=='LM':
        print('# negative:', len(negative_words), '\n# positive:', len(positive_words), '\n# uncertainty:', len(uncertainty_words))
    else:
        print('# negative:', len(negative_words), '\n# positive:', len(positive_words))

    # open discard words: stop words + non-informative words
    discard_words = pd.read_csv('copom_statements/discard_words.csv') # read the select discard words
    discard_words = pd.concat([discard_words['Months'], discard_words['Days'], discard_words['Names'], discard_words['Discards']])# empilha as colunas
    discard_words = list(discard_words.dropna()) # remove nan and transform to a list
    discard_words = discard_words + stopwords.words('english') # add the stop words to the list of discard words
    discard_words = [x.lower() for x in discard_words] # lower case everything

    print('\n# discard_words:', len(discard_words))

    # open the text to be analyzed
    # months = ['jan', 'mar', 'apr', 'jun', 'jul', 'sep', 'oct', 'dec']

    file_names = [
				'2006mar', '2006apr', '2006may', '2006jul', '2006aug', '2006oct', '2006nov',
				'2007jan', '2007mar', '2007apr', '2007jun', '2007jul', '2007sep', '2007oct', '2007dec',
				'2008jan', '2008mar', '2008apr', '2008jun', '2008jul', '2008sep', '2008oct', '2008dec',
				'2009jan', '2009mar', '2009apr', '2009jun', '2009jul', '2009sep', '2009oct', '2009dec',
				'2010jan', '2010mar', '2010apr', '2010jun', '2010jul', '2010sep', '2010oct', '2010dec',
				'2011jan', '2011mar', '2011apr', '2011jun', '2011jul', '2011aug', '2011oct', '2011nov',
				'2012jan', '2012mar', '2012apr', '2012may', '2012jun', '2012aug', '2012oct', '2012nov',
				'2013jan', '2013mar', '2013apr', '2013may', '2013jul', '2013aug', '2013oct', '2013nov',
				'2014jan', '2014feb', '2014apr', '2014may', '2014jul', '2014set', '2014oct', '2014dec',
				'2015jan', '2015mar', '2015apr', '2015jun', '2015jul', '2015set', '2015oct', '2015nov',
				'2016jan', '2016mar', '2016apr', '2016jun', '2016jul', '2016aug', '2016oct', '2016nov',
				'2017jan', '2017feb', '2017apr', '2017may', '2017jul', '2017sep', '2017oct', '2017dec',
				'2018feb', '2018mar', '2018may', '2018jun', '2018aug', '2018sep', '2018oct', '2018dec',
				'2019feb', '2019mar', '2019may', '2019jun', '2019jul', '2019sep', '2019oct', '2019dec',
				'2020feb', '2020mar', '2020may', '2020jun', '2020aug', '2020sep', '2020oct', '2020dec'
			  ]

    statement = 117 #initial statement

#     first_year = 2006
#     last_year = 2021

    root_dir = '/home/cassio/Desktop/paper2WP/raw data/copom_statements/statements/'

    all_statement = [] # empty list to store the name of each statement
    j=1 # just an count

    for yearMonth in file_names:
        file = root_dir + yearMonth + str(statement) + '.txt' # directory of the file to be openned

        text = open(file, 'r').read()    # open the text

        text = text.replace('\xa0', ' ') # substitute spaces

        # time.sleep(.5)
        text = text.replace('\n', ' ')   # substitute lines
        text = re.sub(' +', ' ', text)   # substitute multiple whitespaces

        text = text.replace('<p style="text-align&#58;justify;">', ' ') 
        text = text.replace('<p><span style="text-align&#58;justify;">', ' ')
        text = text.replace('<p>​<span style="text-align&#58;justify;">​', ' ')
        text = text.replace('<p>​<strong style="text-align&#58;justify;">​', ' ')
        text = text.replace('<p class="MsoNormal">', ' ')
        text = text.replace('<p>​</p>', ' ')
        text = text.replace('<p>​</p><p style="margin&#58;0px 0px 10px;text-align&#58;justify;box-sizing&#58;border-box;">', ' ')
        text = text.replace('<p style="margin&#58;0px 0px 10px;text-align&#58;justify;box-sizing&#58;border-box;">', ' ')
        text = text.replace('</p>', ' ')
        text = text.replace('<p>', ' ')



        print(text)
        print()
        text = text.split()              # separete in words

        # for tag in tags:
        #     text = [x.replace(tag, 'In') for x in text]
        #     text = [x.replace(tag, 'The') for x in text]

        text = [x.lower() for x in text] # lower case
        text = [x for x in text if not any(c.isdigit() for c in x)] # exclude all elements of the text that is number

        all_statement.append(yearMonth + str(statement))



        # discard some words

        textFiltered = []
        for word in text:
            if word not in discard_words and len(word)>1:
                textFiltered.append(word)

        # print('     -After excluding stop words and non-informative words, the number of words is', len(textFiltered))

        #*********************************************************
        # classifying the text based on a specialized dictionary
        #*********************************************************

        # positive
        text_positive = []

        for word in textFiltered:
            if word in positive_words:
                text_positive.append(word)
        
        # negative
        text_negative = []

        for word in textFiltered:
            if word in negative_words:
                text_negative.append(word)

        # uncertainty
        if name=='LM':
            text_uncertainty = []
            
            for word in textFiltered:
                if word in uncertainty_words:
                    text_uncertainty.append(word)

        N = len(textFiltered)

        Npositive = len(text_positive)
        Nnegative = len(text_negative)

        if name=='LM':
            Nuncertainty = len(text_uncertainty)

        # save the results in a table

        if name=='LM':
            table = {'# negative': Nnegative, '# positive': Npositive, '# uncertainty': Nuncertainty, 
            'prop negative': Nnegative/N, 'prop positive': Npositive/N, 'prop uncertainty': Nuncertainty/N, 'N':N}
        else:
            table = {'# negative': Nnegative, '# positive': Npositive, 
            'prop negative': Nnegative/N, 'prop positive': Npositive/N,'N': N} 

        table_df =  pd.DataFrame([table])


        if j==1:
            descriptive = table_df
        else:
            descriptive = pd.concat((descriptive, table_df))

        # i+=1
        statement += 1
        j+=1

    # descriptive.index=all_statement

    print('-'*100, '\nDescriptive statistics for the dictionary', name,'\n', '-'*100, '\n\n', descriptive)

descriptive['tone'] = (descriptive["# positive"] -  descriptive["# negative"])/(descriptive["# positive"] +  descriptive["# negative"])

#****************************
# load dates of meetings
#****************************

dates = pd.read_csv('copom_statements/datesOfMeeting.csv') # Load the data and drop all lines with na in any of the columns.

print(dates.head(n=5))
print(dates.tail(n=5))

# index the time series
dates=dates['Datas'].values

dt = []
for date in dates:
    x = '{:}{:}{:}'.format(date[6:], date[3:5], date[:2])
    dt.append(x)


dt = pd.to_datetime(dt, format='%Y%m%d')

print("#"*100)
print(len(dt), len(descriptive))
descriptive.index = dt

# aggregate to monthly 

all_days = pd.date_range(start="2006-01-01", end="2020-12-31", freq="D")
all_days

Dados = pd.DataFrame(np.zeros(len(all_days)), index=all_days, columns=['unnamed'])

matrix = np.zeros((len(Dados),len(descriptive.columns)))

for i in range(len(Dados)):
    if i==1:
        print("Aggregating to monthly frequency...")
    for ii in range(len(descriptive)):
        
        if (descriptive.index[ii]==Dados.index[i]):
            matrix[i,:] = descriptive.values[ii,:]
        

matrix_df = pd.DataFrame(matrix, index=all_days, columns=descriptive.columns)

matrix_df.to_csv("sent.csv")

matrix_df_mensal = matrix_df.reset_index().groupby([matrix_df.index.year, matrix_df.index.month], as_index=False).max().set_index('index')
matrix_df_mensal.index.name=None

final_data = matrix_df_mensal[["prop negative","prop positive","prop uncertainty"]]
# final_data = pd.DataFrame(final_data.iloc[:,:])

print(final_data.columns)
print(len(final_data))
# save
os.chdir("/home/cassio/Desktop/paper2WP")
if not os.path.exists('data'):
    os.makedirs('data')
    

# matrix_df_mensal.to_csv('data/data_sent.csv')
# np.savetxt("data/data_sent_statements.txt", final_data.to_numpy())
np.savetxt("data/data_sent_statements_dates_of_minutes.txt", descriptive.to_numpy())


plt.plot(final_data)
plt.savefig("data/sent_statemets.pdf")

# os.chdir("/home/cassio/Desktop/paper2WP/raw data")
