import re
import math
import collections
#Courtesy of http://stackoverflow.com/questions/6027558/flatten-nested-python-dictionaries-compressing-keys
def flatten(d, parent_key='', sep=' '):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


#General purpose ngram method
def ngram_freqs(in_str, n):
  in_str = in_str.split(' ')
  #Using a dict allows us to get the frequency of word combinations
  #and reference them based on that combination of words.

  output = {}

  #Using range so we can group the inputs together
  include = 0
  if(n == 1):
    include = 1
  for i in range(include,len(in_str)-n+1):
    #in_str[i:i+n] gets us the words from i to i+n, meaning
    #if we want a bigram we'll get two words
    g = ' '.join(in_str[i:i+n])
    output.setdefault(g, 0)
    output[g] += 1
  return output

def unigram(in_str):
  freqs = ngram_freqs(in_str, 1)
  total = len(in_str.split(' ')) - 1

  probs = {word: freqs[word]/total for word in freqs}
  return probs

def ngram(in_str, n):
  if(n == 1):
    return unigram(in_str)

  freqs = ngram_freqs(in_str, n)
  probs = {}
  for key in freqs:
    key_split = key.split(' ')
    base_word = key_split[len(key_split)-1:][0]
    given = key_split[:len(key_split)-1]
    given = ' '.join(given)
    probs.setdefault(given, {})
    probs[given][base_word] = freqs[key]

  for given in probs:
    total = 0
    given_dict = probs[given]
    for base in given_dict:
      total += given_dict[base]
    for base in given_dict:
      given_dict[base] = given_dict[base]/total

  return flatten(probs)


def interpolate(in_str, ngram_list, weights):

  curr_set = 0
  prob = 0
  for ngram in ngram_list:
    in_split = in_str.split(' ')
    curr_string = in_split[len(in_split)-1-curr_set:]
    if(curr_set == 0):
      curr_string = curr_string[0]
    else:
      curr_string = ' '.join(curr_string)
    try:
      prob += ngram[curr_string]*weights[curr_set]
    except KeyError:
      pass
    curr_set += 1
  return -1*math.log(prob, 2)

def replace_unknowns(in_str, k):
  #Getting frequency of words using unigram.
  freqs = ngram_freqs(in_str, 1)
  output = in_str
  for word in freqs:
    if(freqs[word] <= k):
      output = re.sub(r"\s" + word + r"\s", " <unk> ", output)

  return output

def word_perplexity(in_str, ngram_list, n, weights):
  in_str = in_str.split(' ')

  entropy = 0
  #-1 to remove count for <s>
  word_count = len(in_str) - 1
  for i in range(len(in_str)-n+1):
    g = ' '.join(in_str[i:i+n])
    prob = interpolate(g, ngram_list, weights)
    entropy += prob

  entropy = entropy * (1/word_count)
  return 2**entropy

parsed = replace_unknowns("<s> a a , b h </s> <s> a b a i </s> <s> a b , a b a </s> <s> b b , j , b , a </s> <s> a a , a b , b </s> ", 1)

#parsed = "<s> a b a b <unk> </s>"

result = ngram(parsed, 3)
print(result)
for key in result:
  print(key + ': {0}'.format(result[key]))

print('~~~~~~~~~Unigram~~~~~~~~~~~~')
print(ngram(parsed, 1))

all_ngrams = []
for i in range(1, 4):
  all_ngrams.append(ngram(parsed, i))


print("~~~~~~~~~~~~perplexity test~~~~~~~~~~~~~")
print(word_perplexity("<s> a a </s>", all_ngrams, 3, [.3, .3, .4]))
