library(stringr)

load('DictionaryGI.rda')
load('DictionaryHE.rda')
load('DictionaryLM.rda')

path = getwd()

# GI dictionariy: Dictionary with a list of positive and negative words according to the psychological Harvard-IV
#dictionary as used in the General Inquirer software

negGI = data.frame(DictionaryGI$negative)
posGI = data.frame(DictionaryGI$positive)

write.csv(negGI, str_c(path, '/GI/', 'negative.csv'))
write.csv(posGI, str_c(path, '/GI/', 'positive.csv'))

# HE dictionariy: Henry (2008): Are Investors Influenced By How Earnings Press Releases Are Written?, Journal of
# Business Communication, 45:4, 363-407

negHE = data.frame(DictionaryHE$negative)
posHE = data.frame(DictionaryHE$positive)

write.csv(negHE, str_c(path, '/HE/', 'negative.csv'))
write.csv(posHE, str_c(path, '/HE/', 'positive.csv'))

# LM dictionariy: Loughran and McDonald (2011) When is a Liability not a Liability? Textual Analysis, Dictionaries,
# and 10-Ks, Journal of Finance, 66:1, 35-65

negLM = data.frame(DictionaryLM$negative)
posLM = data.frame(DictionaryLM$positive)
uncLM = data.frame(DictionaryLM$uncertainty)

write.csv(negLM, str_c(path, '/LM/', 'negative.csv'))
write.csv(posLM, str_c(path, '/LM/', 'positive.csv'))
write.csv(uncLM, str_c(path, '/LM/', 'uncertainty.csv'))

