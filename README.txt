~~NOTE~~
Perplexity is scored as negative per word entropy, as this apperantely is how
the tests are scored in the sample scoring script for part 3.

Program usage:

  python ngram.py <ngram> <training_file> <dev_file> <test_file>

where ngram is 1, 2, 2s, 3, or 3s. 1 is Unigram, 2 is bigram, 2s is smoothed
bigram using interpolation, etc.

This biggest issue is slow downs with large training sets. The script runs
slowly (up to around 5 minutes) for text files around 1MB large, and larger
than that can take longer than 10 minutes, so I have not completed a test with
training data larger than 1MB.

Note also that pre-processing is minimal so some special characters may throw
it off.


The only libraries used are part of the Python standard library (version
3.5.1).

For part 3 I saved time by doing minimal pre-processing on the files.
Symbols that were causing issues (such as parenthesis, brackets, and
astericks) I replace with an empty char. For commas a space is added before them.
Periods are turned into </s>. The script uses readline to get lines in the
file and adds <s> to the beginning of the training file and test file so that
the machine learning formatted files for part 3 can be tested.

This simplistic method for prepocessing likely costs me in terms of possible
bugs, misrepresented "words" in the ngrams, and overall performance of the
script. 

For smoothing I run my perplexity method around 100 times and adjust weights 
given to each model until an equilibrium is reached. I feel this is naive way
to set the weights and doubt it works as well as other methods, but it gave
me better results than doing it by hand.


There are no other files for Part 3. I piped the results of ngram.py into text
files and used the sample_scoring_script provided in the homework assignment
to test accuracy.
