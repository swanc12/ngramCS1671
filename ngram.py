"""
Author: Colin Swan

Description: Implementation of an ngram model. Takes the type of ngram to use,
the path to a training file, the path to a development file, and the path to
the test file as inputs from the command line. Attempts to use the development
file to determine good weights to use to achieve a lower entropy value.
"""

import re
import math
import collections
import argparse

#Courtesy of http://stackoverflow.com/questions/6027558/flatten-nested-python-dictionaries-compressing-keys
#Used to flatten ngram dictionaries once probabilities have been determined,
#which makes the results a bit easier to work with.
def flatten(d, parent_key='', sep=' '):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


#Gets the number of times each word / word pairs / etc. occur
def ngram_freqs(in_str, n):
  in_str = in_str.strip().split(' ')
  output = {}

  include = 0
  if(n == 1):
    include = 1

  for i in range(include,len(in_str)-n+1):
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
  #Creating a 2D Dict
  #Form: {"wi-2 wi-1": {wi: 2}}
  for key in freqs:
    key_split = key.split(' ')
    base_word = key_split[len(key_split)-1:][0]
    given = key_split[:len(key_split)-1]
    given = ' '.join(given)
    probs.setdefault(given, {})
    probs[given][base_word] = freqs[key]

  #Determing probabilities for words
  #given previous words.
  for given in probs:
    total = 0
    given_dict = probs[given]
    #Counting up total occurances
    #of words after "given"
    for base in given_dict:
      total += given_dict[base]
    #Dividing freq of each word by
    #total occurances of the previous word(s)
    for base in given_dict:
      given_dict[base] = given_dict[base]/total

  return flatten(probs)


def interpolate(in_str, ngram_list, weights):

  #Keeping track of which level of ngram we're at
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
  if(prob <= 0):
    return 0
  return math.log(prob, 2)

def word_perplexity(in_str, ngram_list, n, smooth, weights):
  in_str = in_str.split(' ')

  entropy = 0
  prob = 0
  #-1 to remove count for <s>
  word_count = len(in_str) - 1
  for i in range(len(in_str)-n+1):
    g = ' '.join(in_str[i:i+n])
    if(smooth):
      prob = interpolate(g, ngram_list, weights)
    else:
      try:
        prob = math.log(ngram_list[n-1][g], 2)
      except KeyError:
        prob = 0
        pass
    entropy += prob

  entropy = entropy * (1/word_count)
  return entropy

def replace_unknowns(in_str, k):
  #Getting frequency of words using unigram.
  freqs = ngram_freqs(in_str, 1)
  output = in_str
  for word in freqs:
    if(freqs[word] <= k):
      output = re.sub(r"\s" + word + r"\s", " <unk> ", output)

  return output

def replace_unk_ngram(unigram, in_str):
  output = in_str
  input_list = in_str.split(' ')
  for word in input_list:
    try:
      unigram[word]
    except KeyError:
      output = re.sub(r"\s" + word + r"\s", " <unk> ", output)

  return output

def adjusted_weights(lines, all_ngrams, n, verbose=False):
  weights = []
  if(n == 2):
    weights = [.8, .2]
  elif(n == 3):
    return(adjusted_weights_trigram(lines, all_ngrams, verbose))

  weights_list = []
  for line in lines:
    prev_perp = word_perplexity(line, all_ngrams, n, smoothed, weights)
    weight1 = 30
    weight2 = 70
    weight1up = True
    new_perp = prev_perp

    if(verbose):
      print(line)
      print("Perp before adjusting weights: ")
      print(prev_perp)
    for i in range(100):
      weights = [weight1/100, weight2/100]
      new_perp = word_perplexity(line, all_ngrams, n, True, weights)
      if new_perp > prev_perp and not weight1up and weight2 < 100:
        weight1 -= 1
        weight2 += 1
      elif new_perp < prev_perp and not weight1up and weight1 < 100:
        weight1 += 1
        weight2 -= 1
        weight1up = True
      elif new_perp > prev_perp and weight1up and weight1 < 100:
        weight1 += 1
        weight2 -= 1
      elif weight2 < 100:
        weight1 -= 1
        weight2 += 1
        weight1up = False
      prev_perp = new_perp

    if(verbose):
      print("Unigram weight: {0} Bigram weight: {1}".format(
        weight1,
        weight2,
      print(new_perp)
      ))
    weights_list.append((weight1/100, weight2/100))

  average_weights = []
  if(len(weights_list) > 0):
    weight1_avg = 0
    weight2_avg = 0
    for i in weights_list:
      weight1_avg += i[0]
      weight2_avg += i[1]
    weight1_avg = weight1_avg/len(weights_list)
    weight2_avg = weight2_avg/len(weights_list)
    average_weights.append(weight1_avg)
    average_weights.append(weight2_avg)
  return average_weights

def adjusted_weights_trigram(lines, all_ngrams, verbose):
  weights_list = []
  for line in lines:
    prev_perp = word_perplexity(line, all_ngrams, n, smoothed, [.2, .2, .6])
    weight1 = 20
    weight2 = 20
    weight3 = 60
    weight1up = True
    for i in range(33):
      if(weight2 > 0 and weight3 > 0):
        weight1 = weight1 + 1
        weight2 = weight2 - .5
        weight3 = weight3 - .5
      new_perp = word_perplexity(
          line,
          all_ngrams,
          n,
          smoothed,
          [weight1/100, weight2/100, weight3/100]
      )
      if new_perp > prev_perp:
        continue
      else:
        break

    for i in range(33):
      if(weight1 > 0 and weight3 > 0):
        weight1 = weight1 - .5
        weight2 = weight2 + 1
        weight3 = weight3 - .5
      new_perp = word_perplexity(
          line,
          all_ngrams,
          n,
          smoothed,
          [weight1/100, weight2/100, weight3/100]
      )
      if new_perp > prev_perp:
        continue
      else:
        break

    for i in range(33):
      if(weight1 > 0 and weight2 > 0):
        weight1 = weight1 - .5
        weight2 = weight2 - .5
        weight3 = weight3 + 1
      new_perp = word_perplexity(
          line,
          all_ngrams,
          n,
          smoothed,
          [weight1/100, weight2/100, weight3/100]
      )
      if new_perp > prev_perp:
        continue
      else:
        break
    return [weight1/100, weight2/100, weight3/100]

##Main##
parser = argparse.ArgumentParser()
parser.add_argument(
  'ngram',
  type=str,
  help='Ngram model to use. 1 = unigram, 2 = bigram, 2s = bigram smoothed,'+
    ' 3 = trigram, 3s = trigram smoothed'
)
parser.add_argument('trainfile', type=str, help='Path to training file')
parser.add_argument('devfile', type=str, help='Path to development file')
parser.add_argument('testfile', type=str, help='Path to test file')
args = parser.parse_args()

n = int(args.ngram[0])
smoothed = False
try:
  if(args.ngram[1] == 's'):
    smoothed = True
except Exception:
  pass

parsed = ''
with open(args.trainfile, 'r') as trainfile:
  for line in trainfile:
    temp_parsed = line.strip().replace('.', ' </s>')
    temp_parsed = temp_parsed.replace(',', ' ,')
    temp_parsed = " <s> " + temp_parsed
    temp_parsed = temp_parsed.replace("*", '')
    temp_parsed = temp_parsed.replace("(", '')
    temp_parsed = temp_parsed.replace(")", '')
    temp_parsed = temp_parsed.replace("[", '')
    temp_parsed = temp_parsed.replace("]", '')
    parsed = parsed + temp_parsed

parsed = replace_unknowns(parsed, 1)

all_ngrams = []
for i in range(1, n+1):
  all_ngrams.append(ngram(parsed, i))


lines = []
with open(args.devfile, 'r') as devfile:
  for line in devfile:
    temp_parsed = line.strip()
    temp_parsed = temp_parsed.replace("*", '')
    temp_parsed = temp_parsed.replace("(", '')
    temp_parsed = temp_parsed.replace(")", '')
    temp_parsed = temp_parsed.replace("[", '')
    temp_parsed = temp_parsed.replace("]", '')
    lines.append(replace_unk_ngram(all_ngrams[0], temp_parsed))

new_weights = []
if(smoothed):
  print("~~Adjusted weights~~")
  print(adjusted_weights(lines, all_ngrams, n))
  new_weights = adjusted_weights(lines, all_ngrams, n)

lines = []
with open(args.testfile, 'r') as testfile:
  for line in testfile:
    temp_parsed = line.strip()
    temp_parsed = temp_parsed.replace("*", '')
    temp_parsed = temp_parsed.replace("(", '')
    temp_parsed = temp_parsed.replace(")", '')
    temp_parsed = temp_parsed.replace("[", '')
    temp_parsed = temp_parsed.replace("]", '')
    lines.append(replace_unk_ngram(all_ngrams[0], temp_parsed))

print("Per word perplexity of test file lines: ")
if(smoothed):
  print(new_weights)

for line in lines:
  print(line)
  print(word_perplexity(line, all_ngrams, n, smoothed, new_weights))
