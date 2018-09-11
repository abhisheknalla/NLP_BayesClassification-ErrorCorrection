
# coding: utf-8

# # Tokenization

# In[90]:


from os import listdir
from os.path import isfile, join
import re
import pandas as pd
import glob
import errno


# In[112]:



tokens = {}

delim = "( )"

def tokenize(sentences):
    for sentence in sentences:
        arr = re.split(r':|;|,|-| ',sentence)
        
        for word in arr:
            word = word.strip("'|)|(|>|<|?|[|]|(|)|{|}")
            
            word = word.strip('"')
        
            if word:
                if word.lower() not in tokens:
                    tokens[word.lower()] = 0
            
                tokens[word.lower()]+= 1
                word = word.lower()


# In[113]:


def tokenize_1(sentence): 
    return re.split(r':|;|,|-| ',sentence) # Naive tokeniser


# In[114]:



def tokenize_corpus(cor):
    return [tokenize_1(sentence) for sentence in cor]


# In[115]:


puncts = "('|;|:|-,!)"
caps = "([A-Z])"
smalls =  "([a-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"

def read_content(corpus):
    corpus = corpus.replace('\n',' ')
    
    corpus = re.sub(caps + "[.]" + caps +"[.]","\\1<>\\2<>",corpus)
    corpus = re.sub(" "+ caps +"[.]","\\1<>",corpus)
    corpus = re.sub(caps + "[.]" + caps +"[.]"+ caps +"[.]","\\1<>\\2<>\\3<>",corpus)
    corpus = re.sub(websites,"<>\\1",corpus)
    corpus = re.sub(acronyms +" "+ prefixes,"\\1>< \\2",corpus)
    
    corpus = corpus.replace('?',"?><")
    corpus = corpus.replace('!',"!><")
    corpus = corpus.replace('.',".><")
    corpus = corpus.replace("<>",".")
    corpus = corpus.replace(','," ")
    corpus = corpus.replace("--"," "+"-"+"-"+" ")
    
    sentences = corpus.split("><")
    return sentences[:-1]


# # Correcting the text

# In[ ]:


from ipy_table import *
sorted_tokens = sorted(tokens.items(), key = lambda x: x[1], reverse=True)
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.graph_objs import *
init_notebook_mode(connected=True)

new_text = []

by_new_lines = re.compile("\n+")

def corrected_text(x):
    pre = str('')
    
    x = by_new_lines.split(x)
    
    for line in x:
        line = line.split('   ')
        n = len(line)
        
        if n==3:
            if line[-2] != '-':
                if pre != line[1]:
                    pre = line[1]
                    new_text.append(pre)
            else:
                new_text.append('')
        elif n==1:
            new_text.append(line[0])
    
    return "\n".join(new_text)


# # Class Prediction

# In[117]:


class_corr = {}

class_count = {}

err_class = {}

by_new_lines = re.compile("\n+")

def get_class(x):
    line = by_new_lines.split(x)
    n = len(line)
    for i in range(1, n-3):

        if (line[i-1].split(' ')[0] != '.' and line[i].split(' ')[0] != '.' and line[i+1].split(' ')[0] != '.'):
            
            arr = line[i].split('   ')
            arr_len = len(arr)
            if arr_len == 3:      
                if arr[2] not in class_count:
                    
                    class_count[arr[2]] = {}
                    class_corr[arr[2]] = {}
                    err_class[arr[2]] = 0
                
                err_class[arr[2]] += 1

                if (line[i-1].split('   ')[0], arr[0], line[i+1].split('   ')[0]) not in class_corr[arr[2]]:
                    class_corr[arr[2]][(line[i-1].split('   ')[0], arr[0], line[i+1].split('   ')[0])] = arr[1]
                    
                if (line[i-1].split('   ')[0], arr[0], line[i+1].split('   ')[0]) not in class_count[arr[2]]:
                    class_count[arr[2]][(line[i-1].split('   ')[0], arr[0], line[i+1].split('   ')[0])] = 0
                
                class_count[arr[2]][(line[i-1].split('   ')[0], arr[0], line[i+1].split('   ')[0])] += 1
                


# ## Bigram generator 
# 

# In[118]:



bigrams = {}
bigrams_p = {}

def get_bigrams(corpus):
    for sentence in corpus:
        for index, word in enumerate(sentence):
            word = word.strip("'|)|(|<|>|.|?|!")
            if word:
                word = word.lower()
                
                sentence[index] = word
                
                if index > 0:
                    if sentence[index-1]:
                        
                        prev = sentence[index-1]
                        
                        if prev not in bigrams:
                            bigrams[prev] = {}
                        
                        if word not in bigrams[prev]:
                            bigrams[prev][word] = 0 
                        
                        pair = (sentence[index-1],word)
                        
                        bigrams[prev][word] += 1                        
                        
                        if pair not in bigrams_p:
                            bigrams_p[pair] = 0
                        
                        bigrams_p[pair] += 1
                        


# In[119]:


sorted_bigrams = sorted(bigrams_p.items(), key = lambda x: x[1], reverse=True)

sorted_bigrams = sorted_bigrams[:1000] # RAM problems
#display(make_table(sorted_bigrams[:100]))
#iplot([{"x" : ['_'.join(x) for x in list(zip(*sorted_bigrams[:1000]))[0]], "y": list(zip(*sorted_bigrams[:1000]))[1]}])


# ## Trigram generator

# In[120]:


trigrams = {}

trigrams_t = {}

def get_trigrams(corpus):
    for sentence in corpus:
        
        for index, word in enumerate(sentence):
            word = word.strip("'|)|(|<|>|!|?|. ")
            
            if word:
                word = word.lower()
                sentence[index] = word
                
                if index > 1:
                    if sentence[index-2]:    
                        if index > 0:
                            if sentence[index-1]:
                                pair = (sentence[index-2],sentence[index - 1])
                                
                                if pair not in trigrams:
                                    trigrams[pair] = {}
                                
                                if word not in trigrams[pair]:
                                    trigrams[pair][word] = 0
                
                                tier = (sentence[index-2],sentence[index-1],word)
                                trigrams[pair][word] += 1 
                                
                                if tier not in trigrams_t:
                                    trigrams_t[tier] = 0
                                
                                trigrams_t[tier] += 1
                                
                                pair1 = (sentence[index-2],word)
                
                                if pair1 not in trigrams_t:
                                    trigrams_t[pair1] = 0
                                
                                trigrams_t[pair1] += 1
                                


# In[121]:


sorted_trigrams = sorted(trigrams_t.items(), key = lambda x: x[1], reverse=True)

sorted_trigrams = sorted_trigrams[:1000] # RAM problems
#display(make_table(sorted_trigrams[:100]))


# # Probabilities

# In[122]:


import math

def unigram_probs(tokens):
    new_unigrams = {}
    N = sum(tokens.values())
    
    for word in tokens:
        new_unigrams[word] = round(tokens[word] / float(N), 15)
    
    return new_unigrams


# In[123]:


uprobs = unigram_probs(tokens)
sorted_uprobs = sorted(uprobs.items(), key = lambda x:x[1], reverse=True)
hi_uni = sorted_uprobs[:10]
#display(make_table(hi_uni))
low_uni = sorted_uprobs[:-11:-1]


# In[124]:



def bigram_probs(word):
    N = sum(bigrams[word[0]].values())
    count = 0
    
    if word[1] in bigrams[word[0]]:
        count = bigrams[word[0]][word[1]]
    
    return [round(count / float(N),15),count]


# In[125]:


bg_word = sorted(unigram_probs(bigrams['']).items(),key = lambda x:x[1], reverse=True)
hi_probs = bg_word[:10]
low_probs = bg_word[:-11:-1]
#make_table(hi_probs)


# In[126]:




def trigram_probs(word):
    pair = (word[0],word[1])
    N = sum(trigrams[pair].values())
    count = 0
    
    if word[2] in trigram_probs[pair]:
        count = trigrams[pair][word[2]]
    
    return (round(count / float(N),15),count)


# In[127]:


tg_word = sorted(unigram_probs(trigrams[('','')]).items(),key = lambda x:x[1], reverse=True)
hi_probs = tg_word[:10]
low_probs = tg_word[:-11:-1]
#make_table(hi_probs)


# # Smoothing

# In[128]:



def laplace_smoothing(n_grams):
    n = len(n_grams)
 #   print(n)
    if n==1:
  #      print(n_grams)     
        if n_grams not in unigrams:
            count_value = 1
        else:
            count_value = unigrams[n_grams] + 1
        ans_value = count_value / ((sum(unigrams.values()) + n )* 1.0)  
    if n == 2:
        if n_grams not in bigrams:
            count_value = 1
        else:
            count_value = bigrams[n_grams] + 1
        ans_value = count_value / ((sum(bigrams.values()) + n*(n-1)) * 1.0)
    if n == 3:
        if n not in trigrams:
            count_value = 1
        else:
            count_value = trigrams[n_grams] + 1
        ans_value = count_value / ((sum(bigrams.values()) + n*(n-1)*(n-2)) * 1.0)
    return ans_value


def good_turing(n_grams):
    
    if len(list(n_grams)) == 3:
        if n_grams in trigrams:
#             trigrams[word] + 1
            if trigrams[n_grams] + 1 in freq_class_tri:
                num = (trigrams[n_grams] + 1) * freq_class_tri[trigrams[n_grams] + 1]
            else:
                num = 0
            den = freq_class_tri[trigrams[n_grams]]
            prob = (num/den)/sum(trigrams.values())
        else: 
            count = 1
            if count in freq_class_tri:
                num = (trigrams[n_grams] + 1) * freq_class_tri[trigrams[n_grams] + 1]
            else:
                num = 0
            den = freq_class_tri[trigrams[n_grams]]
            prob = num/den
            
    if len(list(n_grams)) == 2:
        if n_grams in bigrams:
#             trigrams[word] + 1
            if bigrams[n_grams] + 1 in freq_class_bi:
                num = (bigrams[n_grams] + 1) * freq_class_bi[bigrams[n_grams] + 1]
            else:
                num = 0
            print(bigrams[n_grams])
            print(freq_class_bi[bigrams[n_grams]])
            den = freq_class_bi[bigrams[n_grams]]
            
            prob = (num/den)/sum(freq_class_bi.values())
        else: 
            count = 1
            if count in freq_class_bi:
                num = (bigrams[n_grams] + 1) * freq_class_bi[bigrams[n_grams] + 1]
            else:
                num = 0
            den = freq_class_bi[bigrams[n_grams]]
            prob = num/den
            
    return prob

def good_turing_uni(n_grams):
#     print('HERE')
#     print(unigrams)
    if n_grams in unigrams:
        if unigrams[n_grams] + 1 in freq_class_uni:
            prob = ((unigrams[n_grams]+ 1) * freq_class_uni[unigrams[n_grams]] / (freq_class_uni[unigrams[n_grams] ] * 1.0)) / (freq_class_uni[1] * 1.0)
        else:
            prob = 0
    else:
        if 1 in freq_class_uni:
            prob = ((unigrams[n_grams] + 1) * freq_class_uni[unigrams[n_grams]] / ( sum(unigrams.values()) * 1.0))
        else:
            prob = 0

    return prob

def deleted_interpolation(n_grams):
  # perform deleted Interpolation
#     list_grams = list(n_grams)
    n = len(n_grams)
    if n == 1:
        return laplace_smoothing(n_grams)
    if n == 2:
        return 0.7*laplace_smoothing(n_grams) + 0.3*laplace_smoothing((n_grams[1],))
    else:
        return ((0.5)*laplace_smoothing(n_grams) + (0.4)*laplace_smoothing(n_grams[1:3]) + (0.1)*laplace_smoothing((n_grams[2],)))


# In[129]:


train_path = './Data/train.txt'

file = str(train_path)

by_new_lines = re.compile("\n+")

corpus = open(file,'r')

y = corpus.read()

x = corrected_text(y)

x = read_content(x)

get_class(y)

tokenize(x)

cor = tokenize_corpus(x)

get_bigrams(cor)

get_trigrams(cor)


# # Error Detection

# In[131]:


test_path = './Data/dev.txt'


error_dict = {}

x = open(test_path,'r')

x = x.read()

line = by_new_lines.split(x)

for i in range(1,len(line)-3):
    num = 1
    denom = 0
    v = len(tokens)
    if line[i-1].split(' ')[0]!= '.' and line[i].split(' ')[0]!= '.' and line[i+1].split(' ')[0]!= '.':
        
        if (line[i-1], line[i], line[i+1]) in trigrams_t:
            num = num + trigrams_t[(line[i-1],line[i],line[i+1])]
        
        if (line[i-1],line[i+1]) in trigrams_t:
            denom = denom + trigrams_t[(line[i-1],line[i+1])]
        p = (num/(denom+(v*(v-1)*(v-2))))
        
        num = 1
        if (line[i-1],line[i]) in bigrams_p:
            num = num +  bigrams_p[(line[i-1],line[i])]
        denom = 1
        
        if (line[i-1]) in tokens:
            denom = denom + tokens[line[i-1]]      
        p = p + (num/(denom+(v*(v-1))))
        
        if p < 2/((v*(v-1)*(v-2))) + 2/(v*(v-1)) + 2/v:
            if (line[i-1],line[i],line[i+1]) not in error_dict:
                error_dict[(line[i-1],line[i],line[i+1])] = 0
            
            error_dict[(line[i-1],line[i],line[i+1])] += 1
    


# # Writing results

# In[132]:


x = open(test_path,'r')
test_result = 'dev_results.txt'

fd = open(test_result,'w')
x = x.read()
count = 0
output_text = []

line = by_new_lines.split(x)

fd.write('%s\n' %line[0])

line_len = len(line)

for i in range(1, line_len-3):
    
    if line[i-1].split(' ')[0] != '.' and line[i].split(' ')[0] != '.' and line[i+1].split(' ')[0] != '.':
        if (line[i-1], line[i], line[i+1]) in error_dict:
            
            word = ''
            clas = ''
            corr = ''
            
            ans = -1
            l = -1
            
            for cc in class_count:
                if (line[i-1],line[i],line[i+1]) in class_count[cc]:
                        prob1 = round((class_count[cc][(line[i-1],line[i],line[i+1])])/(err_class[cc]),15)
                        prob2 = round((err_class[cc])/(sum(err_class.values())),15)
                        
                        l = prob1*prob2
                        
                if ans < l:
                    ans = l
                    word = line[i]
                    clas = cc
                    corr = class_corr[cc][(line[i-1],line[i],line[i+1])]
            
            if word:
                fd.write('%s   %s   %s\n' %(word,corr,clas))
            
            else:
                fd.write('%s\n' %(line[i]))
        
        else:
            fd.write('%s\n' %(line[i]))
    
    else:
        fd.write('%s\n' %(line[i]))

fd.close()      


# In[138]:


file = test_result

test_correct = './Data/dev_correction_results.txt'

file1 = test_correct

x = open(file,'r')
x = x.read()

y = open(file1,'r')
y = y.read()

count = 0

x = by_new_lines.split(x)
y = by_new_lines.split(y)

for i in range(len(x)):
    if len(x[i].split('   ')) == len(y[i].split('   ')) and len(x[i].split('   '))!= 1:
        print(i, x[i], y[i])
        count += 1

print(count)


# In[137]:


print(count)

