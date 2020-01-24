

import sys 
import re
import math
from fractions import Fraction

# Open the file in read mode 
#text = open(sys.argv[1], "r") 

train = "1b_benchmark.train.tokens"
dev = "1b_benchmark.dev.tokens"
test = "1b_benchmark.test.tokens"

preprocessedTrain = "train.preprocessed"
preprocessedDev = "dev.preprocessed"
preprocessedTest = "test.preprocessed"

dashLine = "------------------------------------------------------------------------"

def preprocessData(trainPath, devPath, testPath):
    trainFile = open(trainPath, "r")

    freq_dict = dict()
    corpusSize = 0
    count = 0

    for line in trainFile:
        count = count + 1
        line = '<START> ' + line + ' <STOP>'
        tokens = line.split(" ")
        corpusSize = len(tokens) - 1

        for token in tokens:
            token = token.strip()

            # if token is already in dictionary then increment the count
            if token in freq_dict:
                freq_dict[token] = freq_dict[token] + 1
            # else initialize that token's key with value 1
            elif token != '<START>':
                freq_dict[token] = 1

    vocabulary = {k:v for (k,v) in freq_dict.items() if v >= 3}
    vocabulary['<UNK>'] = 0

    unkwords = set()

    for k, v in freq_dict.items():
        if(v < 3): 
            vocabulary['<UNK>'] = vocabulary['<UNK>'] + freq_dict[k]
            unkwords.add(k)

    ppTrain = open(preprocessedTrain, "w")
    trainFile.close()
    trainFile = open(trainPath, "r")

    for line in trainFile:
        newLine = line.strip().split()

        tokens = line.strip().split(" ")

        i = 0

        while i < len(tokens):
            tokens[i] = tokens[i].strip()
            if tokens[i] in unkwords:
                newLine[i] = '<UNK>'

            i += 1

        for token in newLine:
            ppTrain.write(token + " ")

        ppTrain.write("\n")

    ppTrain.close()
    trainFile.close()



    devFile = open(devPath, "r")
    ppDev = open(preprocessedDev, "w")

    for line in devFile:
        newLine = line.strip().split()

        tokens = line.strip().split(" ")

        i = 0

        while i < len(tokens):
            tokens[i] = tokens[i].strip()
            if vocabulary.get(tokens[i], 0) == 0:
                newLine[i] = '<UNK>'

            i += 1

        for token in newLine:
            ppDev.write(token + " ")

        ppDev.write("\n")

    devFile.close()
    ppDev.close()


    testFile = open(testPath, "r")
    ppTest = open(preprocessedTest, "w")

    for line in testFile:
        newLine = line.strip().split()

        tokens = line.strip().split(" ")

        i = 0

        while i < len(tokens):
            tokens[i] = tokens[i].strip()
            if vocabulary.get(tokens[i], 0) == 0:
                newLine[i] = '<UNK>'

            i += 1

        for token in newLine:
            ppTest.write(token + " ")

        ppTest.write("\n")

    testFile.close()
    ppTest.close()

  #  vocabulary['<START>'] = count
  #  vocabulary['<STOP>'] = count

    return (vocabulary, corpusSize)

# global variable for vocabulary and corpusSize


class UnigramModel:
    def __init__(self, vocab, cS):
        print("Constructing Unigram Model...")
        textFile = open(preprocessedTrain, "r")
        self.freq_dict = dict()
        self.corpusSize = 0

        for line in textFile:
            # append stop token to EOLs
            line = line + ' <STOP>'
            # split the line into tokens
            tokens = line.split(" ")
            # iterate through each token
            for token in tokens:
                self.corpusSize += 1
                # remove whitespace and \n characters
                token = token.strip()
                # if token is already in dictionary then increment the count
                if token in self.freq_dict:
                    self.freq_dict[token] = self.freq_dict[token] + 1
                # else initialize that token's key with value 1
                else:
                    self.freq_dict[token] = 1

        self.freq_dict = self.replaceUNKs(self.freq_dict)
        textFile.close()
        print("Unigram model constructed!")

    # this function replaces all keys with < 3 frequency with 'unk'
    def replaceUNKs(self, idict):
        fdict = {k:v for (k,v) in idict.items() if v >= 3}
        fdict['unk'] = 0
        for k, v in idict.items():
            if(v < 3): fdict['<UNK>'] = fdict['<UNK>'] + idict[k]
        return fdict

    def calcTokenProb(self, token):
        #The numerator is the frequency of the specified token, return 0 if not found in dict
        unigram_numerator = self.freq_dict.get(token, 0)

        #If frequency is 0, the word is unkown
        if unigram_numerator == 0:
            unigram_numerator = self.freq_dict.get('<UNK>', 0)

        #The denominator is just the size of the corpus
        unigram_denominator = self.corpusSize
        #print("Pr(" + token + "):")
        #print("numerator = ", unigram_numerator, "denominator = ", unigram_denominator)
        prob = Fraction(unigram_numerator, unigram_denominator)

        return prob

    def calcSentenceProb(self, sentence):

        #Return 0 for empty sentence
        if not sentence:
            return 0

        prob = Fraction(1)
        #The MLE for a sentence is the product of the MLEs of its constituent tokens
        for token in sentence:
            prob = prob * self.calcTokenProb(token)

        return prob

    #Takes a list of sentences and calculates the perplexity of the model given those senetences
    def calcPerplexityOLD(self, sentences):
        logSum = 0

        #Keep track of how many tokens are in the sample we are testing
        sampleSize = 0

        #Find the log probability of each sentence and sum them all up
        for sentence in sentences:
            sampleSize += len(sentence)
            sentenceProb = self.calcSentenceProb(sentence)

            #Log is undefined if input is not > 0
            if sentenceProb.numerator > 0:
                logProb = math.log(sentenceProb.numerator, 2) - math.log(sentenceProb.denominator, 2)
                logSum += logProb
            #Here we assume Pr = 0, so perplexity is infinite, return -1 to signify this
            else:
               print("Sentence:" + str(sentence))
               return -1

        #To get perplexity, multiply this sum by the negative reciprocal 
        #of sample size and exponentiate it base 2
        logSum = logSum * (-1 / float(sampleSize))
        perplexity = 2 ** logSum

        return perplexity

    def calcPerplexity(self, sentences):
        sampleLogSum = 0

        #Keep track of how many tokens are in the sample we are testing
        sampleSize = 0

        #Find the log probability of each sentence and sum them all up
        for sentence in sentences:
            sampleSize += len(sentence)

            sentenceLogSum = 0

            #The log probability of each sentence is the sum of the log probabilities
            #of its constituent tokens
            for token in sentence:
                tokenProb = self.calcTokenProb(token)

                if tokenProb > 0:
                    sentenceLogSum += math.log(tokenProb.numerator, 2) - math.log(tokenProb.denominator, 2)
                else:
                    print("Token: " + token)
                    return -1

            sampleLogSum += sentenceLogSum

        #To get perplexity, multiply this sum by the negative reciprocal 
        #of sample size and exponentiate it base 2
        sampleLogSum = sampleLogSum * (-1 / float(sampleSize))
        perplexity = 2 ** sampleLogSum

        return perplexity


    #Uses a test file and calculates perplexity based on that sample
    def testModel(self, test):
        testFile = open(test, "r")
        print("Calculating perplexity of " + test)

        sentences = []

        for line in testFile:
            # append stop token to EOLs
            line = line.strip() + ' <STOP>'

            # split the line into tokens and strip whitespace
            sentence = [token.strip() for token in line.split(" ")]

            sentences.append(sentence)

        testFile.close()

        perplexity = self.calcPerplexity(sentences)
        if perplexity == -1: return 'inf'
        else: return perplexity

class BigramModel:
    def __init__(self, vocab, cS):
    #    self.uni = unigramModel
        print("Constructing Bigram Model...")
        textFile = open(preprocessedTrain, "r")
        self.freq_dict = dict()
        self.corpusSize = 0
        for line in textFile:
             # append stop token to EOLs
            line = '<START> ' + line.strip() + ' <STOP>'
             # split the line into tokens
            tokens = line.split(" ")
            bigramList = self.createBigrams(tokens)
            for bigram in bigramList:
                #if bigram == ('Long','before'): print(str(bigram) + " BIGRAM")
              #  print(str(bigram))
                if bigram in self.freq_dict:
                     self.freq_dict[bigram] = self.freq_dict[bigram] + 1
                 # else initialize that token's key with value 1
                else:
                     self.freq_dict[bigram] = 1
#        
        print("Bigram model constructed!") 
      #  self.replaceUNKs(self.freq_dict, self.uni.unkList)

    # BigramModel('1b_benchmark.test.tokens', UnigramModel('1b_benchmark.test.tokens'))

    #Returns a new bigram dictionary with unked words
    def replaceUNKs(self, idict, unks):
        firstWord = ""
        secondWord = ""
        fdict = dict()

        #Loop through the entire given dicts
        for (fst, snd) in idict:
            #Loop through the list of words that are to be converted to unk
            for unk in unks:

                #IF either word in the bigram is equal to any of the unked words, convert it to unk
                if fst == unk:
                    firstWord = 'unk'

                if snd == unk:
                    secondWord = 'unk'

            #If either word is still empty, it is the same
            if firstWord == "":
                firstWord = fst

            if secondWord == "":
                secondWord = snd

            #Check if the bigram is already in the dictionary before adding it
            #This should only happen for bigrams containing unk
            if (firstWord, secondWord) in fdict:
                fdict[(firstWord, secondWord)] += idict[(fst, snd)]
            else:
                fdict[(firstWord, secondWord)] = idict[(fst, snd)]
        print(fdict)
        return fdict

    def createBigrams(self, sentence):
        if len(sentence) < 2:
            return []

        i = 1
        bigramList = []

        while i < len(sentence):
            bigramList.append((sentence[i - 1], sentence[i]))
            i = i + 1

        return bigramList

    def calcTokenProb(self, tt):
        bigram_numerator = self.freq_dict.get((tt[0],tt[1]), 0)
        bigram_denominator = vocab.get(tt[0], 0)
        if bigram_numerator == 0 or bigram_denominator == 0:
            print("bigram num = " + str(bigram_numerator))
            print("bigram den = " + str(bigram_denominator))
            return 0
        else: return Fraction(bigram_numerator, bigram_denominator)

    def calcPerplexity(self, sentences):
        sampleLogSum = 0

        #Keep track of how many tokens are in the sample we are testing
        sampleSize = 0

        #Find the log probability of each sentence and sum them all up
        for sentence in sentences:
            sampleSize += len(sentence)

            sentenceLogSum = 0

            for i in range(1, len(sentence) - 1):
                if sentence[i-1] == '<START>': continue
                else: tokenProb = self.calcTokenProb((sentence[i-1], sentence[i]))
              #  print(tokenProb)         
                if tokenProb > 0:
                    sentenceLogSum += math.log(tokenProb.numerator, 2) - math.log(tokenProb.denominator, 2)
                else:
                    print("Token: " + str((sentence[i-1], sentence[i])))
                    return -1

            sampleLogSum += sentenceLogSum
        #To get perplexity, multiply this sum by the negative reciprocal 
        #of sample size and exponentiate it base 2
        sampleLogSum = sampleLogSum * (-1 / float(sampleSize))
        perplexity = 2 ** sampleLogSum

        return perplexity

    def testModel(self, test):
        testFile = open(test, "r")
        print("Calculating perplexity of " + test)

        sentences = []

        for line in testFile:
            # append stop token to EOLs
            line = '<START> ' + line.strip() + ' <STOP>'

            # split the line into tokens and strip whitespace
            sentence = [token.strip() for token in line.split(" ")]

            sentences.append(sentence)

        testFile.close()
        perplexity = self.calcPerplexity(sentences)
        if perplexity == -1: return 'inf'
        else: return perplexity


class TrigramModel:
    def __init__(self, vocab, cS):
        print("Constructing Trigram Model...")
      #  bigram = BigramModel(vocab,cS)
        textFile = open(preprocessedTrain, "r")
        self.freq_dict = dict()
        self.corpusSize = 0

        for line in textFile:
            line = '<START> ' + line.strip() + ' <STOP>'
            tokens = line.split(" ")
            trigramList = self.createTrigrams(tokens)

            for trigram in trigramList:
                if trigram in self.freq_dict:
                    self.freq_dict[trigram] = self.freq_dict[trigram] + 1
                 # else initialize that token's key with value 1
                else:
                     self.freq_dict[trigram] = 1
        print("Trigram model constructed!")

    def createTrigrams(self, sentence):
        if len(sentence) < 2:
            return []
        i = 2
        trigramList = []

        while i < len(sentence):
            trigramList.append((sentence[i - 2], sentence[i - 1], sentence[i]))
            i = i + 1

        return trigramList

    def calcTokenProb(self, tt):
        trigram_numerator = self.freq_dict.get((tt[0],tt[1],tt[2]), 0)
        trigram_denominator = b.freq_dict.get((tt[0],tt[1]), 0)
        if trigram_numerator == 0 or trigram_denominator == 0:
            print("trigram num = " + str(trigram_numerator))
            print("trigram den = " + str(trigram_denominator))
            return 0
        else: return Fraction(trigram_numerator, trigram_denominator)

    def calcPerplexity(self, sentences):
        sampleLogSum = 0

        #Keep track of how many tokens are in the sample we are testing
        sampleSize = 0

        #Find the log probability of each sentence and sum them all up
        for sentence in sentences:
            sampleSize += len(sentence)

            sentenceLogSum = 0

            for i in range(2, len(sentence) - 1):
                tokenProb = self.calcTokenProb((sentence[i-2], sentence[i-1], sentence[i]))
              #  print(tokenProb)         
                if tokenProb > 0:
                    sentenceLogSum += math.log(tokenProb.numerator, 2) - math.log(tokenProb.denominator, 2)
                else:
                    print("Token: " + str((sentence[i-2], sentence[i-1], sentence[i])))
                    return -1

            sampleLogSum += sentenceLogSum
        #To get perplexity, multiply this sum by the negative reciprocal 
        #of sample size and exponentiate it base 2
        sampleLogSum = sampleLogSum * (-1 / float(sampleSize))
        perplexity = 2 ** sampleLogSum

        return perplexity

    def testModel(self, test):
        testFile = open(test, "r")
        print("Calculating perplexity of " + test)

        sentences = []

        for line in testFile:
            # append stop token to EOLs
            line = '<START> ' + line.strip() + ' <STOP>'

            # split the line into tokens and strip whitespace
            sentence = [token.strip() for token in line.split(" ")]

            sentences.append(sentence)

        testFile.close()

        perplexity = self.calcPerplexity(sentences)
        if perplexity == -1: return 'inf'
        else: return perplexity
        
def main():

    path = sys.argv[1] + "/"

    preprocessData(path + train, path + dev, path + test)

    """
    print("\n" + dashLine)
    uni = UnigramModel(path + train)
    print(dashLine + "\n")
    
    print("\n" + dashLine)
    print("Unigram train perplexity: " + str(uni.testModel(path + train)))
    print("")
    print("Unigram dev perplexity: " + str(uni.testModel(path + dev)))
    print("")
    print("Unigram test perplexity: " + str(uni.testModel(path + test)))
    print(dashLine + "\n")
    """

    # for testing purposes
(vocab, cS) = preprocessData('1b_benchmark.train.tokens','1b_benchmark.dev.tokens','1b_benchmark.test.tokens')
u = UnigramModel(vocab,cS)
b = BigramModel(vocab, cS)
t = TrigramModel(vocab,cS)


sample = '<START> Long before the advent of e-commerce , Wal-Mart s founder Sam Walton set out his vision for a successful retail operation :  We let folks know we re interested in them and that they re vital to us--  cause they are ,  he said . <STOP>'
sample = sample.strip().split(" ")


if __name__ == '__main__':
    main()
