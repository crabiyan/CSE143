import sys 
import re
import math
from fractions import Fraction

train = "1b_benchmark.train.tokens"
dev = "1b_benchmark.dev.tokens"
test = "1b_benchmark.test.tokens"

preprocessedTrain = "train.preprocessed"
preprocessedDev = "dev.preprocessed"
preprocessedTest = "test.preprocessed"

dashLine = "------------------------------------------------------------------------"

#Takes paths to train, dev, and test data as arguments
#Function builds a vocabulary from the training data and creates new files
#identical to the originals except with UNKed words
#Returns the vocabulary and the size of the training corpus
def preprocessData(trainPath, devPath, testPath):

    print("Preprocessing train data...")

    trainFile = open(trainPath, "r")

    freq_dict = dict()
    corpusSize = 0

    #Go through the train file and build a vocabulary 
    for line in trainFile:

        line = '<START> ' + line.strip() + ' <STOP>'
        tokens = line.split()
        corpusSize += len(tokens) - 1

        for token in tokens:
            token = token.strip()

            # if token is already in dictionary then increment the count
            if token in freq_dict:
                freq_dict[token] = freq_dict[token] + 1
            # else initialize that token's key with value 1
            elif token != '<START>':
                freq_dict[token] = 1

    #UNK any words that appear less than 3 times
    #Store any unked words in unkwords for later
    vocabulary = {k:v for (k,v) in freq_dict.items() if v >= 3}
    vocabulary['<UNK>'] = 0

    unkwords = set()

    for k, v in freq_dict.items():
        if(v < 3): 
            vocabulary['<UNK>'] = vocabulary['<UNK>'] + freq_dict[k]
            unkwords.add(k)


    #Open ppTrain to write to
    #Reopen trainfile to start from the beginning
    ppTrain = open(preprocessedTrain, "w")
    trainFile.close()
    trainFile = open(trainPath, "r")

    #Go through all three files and unk any words that need to be
    #Write these to new files
    for line in trainFile:
        newLine = line.strip().split()

        tokens = line.strip().split()

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

    print("Train data preprocessed!")
    print("Preprocessing dev data...")

    devFile = open(devPath, "r")
    ppDev = open(preprocessedDev, "w")

    for line in devFile:
        newLine = line.strip().split()

        tokens = line.strip().split()

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

    print("Dev data preprocessed!")
    print("Preprocessing test data...")

    testFile = open(testPath, "r")
    ppTest = open(preprocessedTest, "w")

    for line in testFile:
        newLine = line.strip().split()

        tokens = line.strip().split()

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

    print("Test data preprocessed!")

    return (vocabulary, corpusSize)

class UnigramModel:
    def __init__(self, vocab, corpusSize):
        print("Constructing Unigram Model...")

        self.unigramFrequency = vocab
        self.corpusSize = corpusSize

        print("Unigram model constructed!")

    #Calculate the probability of a given unigram
    def calcTokenProb(self, token):
        #The numerator is the frequency of the specified token, return 0 if not found in dict
        unigram_numerator = self.unigramFrequency.get(token, 0)

        #If frequency is 0, the word is unkown
        if unigram_numerator == 0:
            unigram_numerator = self.unigramFrequency.get('<UNK>', 0)

        #The denominator is just the size of the corpus
        unigram_denominator = self.corpusSize
        prob = Fraction(unigram_numerator, unigram_denominator)

        return prob

    #Calculate the perplexity of the model on a list of sentences
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
            sentence = [token.strip() for token in line.split()]

            sentences.append(sentence)

        testFile.close()

        return self.calcPerplexity(sentences)


class BigramModel:
    def __init__(self, vocab, cS):
        print("Constructing Bigram Model...")
        textFile = open(preprocessedTrain, "r")
        self.freq_dict = dict()
        self.corpusSize = 0
        self.vocab = vocab
        for line in textFile:
             # append stop token to EOLs
            line = '<START> ' + line.strip() + ' <STOP>'
             # split the line into tokens
            tokens = line.split()
            bigramList = self.createBigrams(tokens)
            for bigram in bigramList:
                if bigram in self.freq_dict:
                     self.freq_dict[bigram] = self.freq_dict[bigram] + 1
                 # else initialize that token's key with value 1
                else:
                     self.freq_dict[bigram] = 1

        print("Bigram model constructed!") 

    #Function that takes a sentence as a list of words and returns a list of
    #2-tuples representing the constituent bigrams of the sentence
    def createBigrams(self, sentence):
        if len(sentence) < 2:
            return []

        i = 1
        bigramList = []

        while i < len(sentence):
            bigramList.append((sentence[i - 1], sentence[i]))
            i = i + 1

        return bigramList

    #Calculate the probability of a given bigram
    def calcTokenProb(self, tt):
        bigram_numerator = self.freq_dict.get((tt[0],tt[1]), 0)
        if tt[0] == '<START>': bigram_denominator = self.vocab.get('<STOP>', 0)
        else: bigram_denominator = self.vocab.get(tt[0], 0)
        if bigram_numerator == 0 or bigram_denominator == 0:
            #print("bigram num = " + str(bigram_numerator))
            #print("bigram den = " + str(bigram_denominator))
            return 0
        else: return Fraction(bigram_numerator, bigram_denominator)

    #Calculate the perplexity of the model on a list of sentences
    def calcPerplexity(self, sentences):
        sampleLogSum = 0

        #Keep track of how many tokens are in the sample we are testing
        sampleSize = 0

        #Find the log probability of each sentence and sum them all up
        for sentence in sentences:
            sampleSize += len(sentence) - 1

            sentenceLogSum = 0

            for i in range(1, len(sentence)):
                tokenProb = self.calcTokenProb((sentence[i-1], sentence[i]))
              #  print(tokenProb)         
                if tokenProb > 0:
                    sentenceLogSum += math.log(tokenProb.numerator, 2) - math.log(tokenProb.denominator, 2)
                else:
                    #print("Token: " + str((sentence[i-1], sentence[i])))
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
            line = '<START> ' + line.strip() + ' <STOP>'

            # split the line into tokens and strip whitespace
            sentence = [token.strip() for token in line.split(" ")]

            sentences.append(sentence)

        testFile.close()
        perplexity = self.calcPerplexity(sentences)
        if perplexity == -1: return 'INFINITY'
        else: return perplexity

class TrigramModel:
    def __init__(self, bigram, cS):
        print("Constructing Trigram Model...")
        textFile = open(preprocessedTrain, "r")
        self.freq_dict = dict()
        self.corpusSize = 0
        self.bigram = bigram

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

    #Function that takes a sentence as a list of words and returns a list of
    #3-tuples representing the constituent trigrams of the sentence
    def createTrigrams(self, sentence):
        if len(sentence) < 3:
            return []

        i = 2
        trigramList = []

        while i < len(sentence):
            trigramList.append((sentence[i - 2], sentence[i - 1], sentence[i]))
            i = i + 1

        return trigramList

    #Calculate the probability of a given trigram
    def calcTokenProb(self, tt):
        if len(tt) == 3:
            trigram_numerator = self.freq_dict.get((tt[0],tt[1],tt[2]), 0)
            trigram_denominator = self.bigram.freq_dict.get((tt[0],tt[1]), 0)
            if trigram_numerator == 0 or trigram_denominator == 0:
                #print("trigram num = " + str(trigram_numerator))
                #print("trigram den = " + str(trigram_denominator))
                return 0
            else: return Fraction(trigram_numerator, trigram_denominator)
        elif len(tt) == 2:
            bigram_numerator = self.bigram.freq_dict.get((tt[0],tt[1]), 0)
            if tt[0] == '<START>': bigram_denominator = self.bigram.vocab.get('<STOP>', 0)
            else: bigram_denominator = self.bigram.vocab.get(tt[0], 0)
            if bigram_numerator == 0 or bigram_denominator == 0:
                #print("Trigram (bigram) num = " + str(bigram_numerator))
                #print("Trigram (bigram) den = " + str(bigram_denominator))
                return 0
            else: return Fraction(bigram_numerator, bigram_denominator)

    #Calculate the perplexity of the model on a list of sentences
    def calcPerplexity(self, sentences):
        sampleLogSum = 0

        #Keep track of how many tokens are in the sample we are testing
        sampleSize = 0

        #Find the log probability of each sentence and sum them all up
        for sentence in sentences:
            sampleSize += len(sentence) - 1

            sentenceLogSum = 0

            firstBigramProb = self.calcTokenProb((sentence[0], sentence[1]))

            if firstBigramProb > 0:
                sentenceLogSum += math.log(firstBigramProb.numerator, 2) - math.log(firstBigramProb.denominator, 2)
            else:
                #print("Token: " + str((sentence[0], sentence[1])))
                return -1

            for i in range(2, len(sentence)):
                tokenProb = self.calcTokenProb((sentence[i-2], sentence[i-1], sentence[i]))
              #  print(tokenProb)         
                if tokenProb > 0:
                    sentenceLogSum += math.log(tokenProb.numerator, 2) - math.log(tokenProb.denominator, 2)
                else:
                    #print("Token: " + str((sentence[i-2], sentence[i-1], sentence[i])))
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
            line = '<START> ' + line.strip() + ' <STOP>'

            # split the line into tokens and strip whitespace
            sentence = [token.strip() for token in line.split(" ")]

            sentences.append(sentence)

        testFile.close()

        perplexity = self.calcPerplexity(sentences)
        if perplexity == -1: return 'INFINITY'
        else: return perplexity

    

class SmoothedModel:
    def __init__(self, uni, bi, tri, l1, l2, l3):
        self.unigram = uni
        self.bigram = bi
        self.trigram = tri
        self.lambda1 = Fraction(l1)
        self.lambda2 = Fraction(l2)
        self.lambda3 = Fraction(l3)

    def setLambdas(self, l1, l2, l3):
        self.lambda1 = Fraction(l1)
        self.lambda2 = Fraction(l2)
        self.lambda3 = Fraction(l3)

    def calcTokenProb(self, ngram):
        #ngram is usually a trigram, but is sometimes a bigram
        tupleSize = len(ngram)

        uniProb = self.unigram.calcTokenProb(ngram[tupleSize - 1])
        biProb = self.bigram.calcTokenProb((ngram[tupleSize - 2], ngram[tupleSize - 1]))
        triProb = self.trigram.calcTokenProb(ngram)

        return (uniProb*self.lambda1 + biProb*self.lambda2 + triProb*self.lambda3)

    #Calculate the perplexity of the model on a list of sentences
    def calcPerplexity(self, sentences):
        sampleLogSum = 0

        #Keep track of how many tokens are in the sample we are testing
        sampleSize = 0

        #Find the log probability of each sentence and sum them all up
        for sentence in sentences:
            sampleSize += len(sentence) - 1

            sentenceLogSum = 0

            firstBigramProb = self.calcTokenProb((sentence[0], sentence[1]))

            if firstBigramProb > 0:
                sentenceLogSum += math.log(firstBigramProb.numerator, 2) - math.log(firstBigramProb.denominator, 2)
            else:
                #print("Token: " + str((sentence[0], sentence[1])))
                return -1

            for i in range(2, len(sentence)):
                tokenProb = self.calcTokenProb((sentence[i-2], sentence[i-1], sentence[i]))
              #  print(tokenProb)         
                if tokenProb > 0:
                    sentenceLogSum += math.log(tokenProb.numerator, 2) - math.log(tokenProb.denominator, 2)
                else:
                    #print("Token: " + str((sentence[i-2], sentence[i-1], sentence[i])))
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
            line = '<START> ' + line.strip() + ' <STOP>'

            # split the line into tokens and strip whitespace
            sentence = [token.strip() for token in line.split(" ")]

            sentences.append(sentence)

        testFile.close()

        perplexity = self.calcPerplexity(sentences)
        if perplexity == -1: return 'INFINITY'
        else: return perplexity
'''
path = sys.argv[1] + "/"
(vocabulary, corpusSize) = preprocessData(path + train, path + dev, path + test)
uni = UnigramModel(vocabulary, corpusSize)
bi = BigramModel(vocabulary, corpusSize)
tri = TrigramModel(bi, corpusSize)
smooth = SmoothedModel(uni, bi, tri, 0.1, 0.3, 0.6)
'''
def main():

    path = sys.argv[1] + "/"

    #Preprocess the data
    print("\n" + dashLine)
    (vocabulary, corpusSize) = preprocessData(path + train, path + dev, path + test)
    print(dashLine + "\n")

    #Build our n-gram models
    print("\n" + dashLine)
    uni = UnigramModel(vocabulary, corpusSize)
    bi = BigramModel(vocabulary, corpusSize)
    tri = TrigramModel(bi, corpusSize)
    print(dashLine + "\n")

    #Test the perplexity of the unigram model
    print("\n" + "Unigram Model")
    print(dashLine)
    print("Unigram train perplexity: " + str(uni.testModel(preprocessedTrain)))
    print("")
    print("Unigram dev perplexity: " + str(uni.testModel(preprocessedDev)))
    print("")
    print("Unigram test perplexity: " + str(uni.testModel(preprocessedTest)))
    print(dashLine + "\n")

    #Test the perplexity of the bigram model
    print("\n" + "Bigram Model")
    print(dashLine)
    print("Bigram train perplexity: " + str(bi.testModel(preprocessedTrain)))
    print("")
    print("Bigram dev perplexity: " + str(bi.testModel(preprocessedDev)))
    print("")
    print("Bigram test perplexity: " + str(bi.testModel(preprocessedTest)))
    print(dashLine + "\n")

    #Test the perplexity of the bigram model
    print("\n" + "Trigram Model")
    print(dashLine)
    print("Trigram train perplexity: " + str(tri.testModel(preprocessedTrain)))
    print("")
    print("Trigram dev perplexity: " + str(tri.testModel(preprocessedDev)))
    print("")
    print("Trigram test perplexity: " + str(tri.testModel(preprocessedTest)))
    print(dashLine + "\n")

    smooth = SmoothedModel(uni, bi, tri, 0.1, 0.3, 0.6)
    
    #Test the perplexity of the smoothed model
    print("\n" + "Smoothed Model (0.1, 0.3, 0.6): This may take a moment...")
    print(dashLine)
    print("Smoothed train perplexity: " + str(smooth.testModel(preprocessedTrain)))
    print("")
    print("Smoothed dev perplexity: " + str(smooth.testModel(preprocessedDev)))
    print("")
    print("Smoothed test perplexity: " + str(smooth.testModel(preprocessedTest)))
    print(dashLine + "\n")

    #Change the lambda coefficients
    smooth.setLambdas(0.33, 0.33, 0.34)

    #Test the perplexity of the smoothed model
    print("\n" + "Smoothed Model (0.33, 0.33, 0.34): This may take a moment...")
    print(dashLine)
    print("Smoothed train perplexity: " + str(smooth.testModel(preprocessedTrain)))
    print("")
    print("Smoothed dev perplexity: " + str(smooth.testModel(preprocessedDev)))
    print("")
    print("Smoothed test perplexity: " + str(smooth.testModel(preprocessedTest)))
    print(dashLine + "\n")

    #Change the lambda coefficients
    smooth.setLambdas(0.15, 0.15, 0.7)

    #Test the perplexity of the smoothed model
    print("\n" + "Smoothed Model (0.7, 0.15, 0.15): This may take a moment...")
    print(dashLine)
    print("Smoothed train perplexity: " + str(smooth.testModel(preprocessedTrain)))
    print("")
    print("Smoothed dev perplexity: " + str(smooth.testModel(preprocessedDev)))
    print("")
    print("Smoothed test perplexity: " + str(smooth.testModel(preprocessedTest)))
    print(dashLine + "\n")

    #Change the lambda coefficients
    smooth.setLambdas(0.1, 0, 0.9)

    #Test the perplexity of the smoothed model
    print("\n" + "Smoothed Model (0.1, 0, 0.9): This may take a moment...")
    print(dashLine)
    print("Smoothed train perplexity: " + str(smooth.testModel(preprocessedTrain)))
    print("")
    print("Smoothed dev perplexity: " + str(smooth.testModel(preprocessedDev)))
    print("")
    print("Smoothed test perplexity: " + str(smooth.testModel(preprocessedTest)))
    print(dashLine + "\n")

    #Change the lambda coefficients
    smooth.setLambdas(0.9, 0, 0.1)

    #Test the perplexity of the smoothed model
    print("\n" + "Smoothed Model (0.9, 0, 0.1): This may take a moment...")
    print(dashLine)
    print("Smoothed train perplexity: " + str(smooth.testModel(preprocessedTrain)))
    print("")
    print("Smoothed dev perplexity: " + str(smooth.testModel(preprocessedDev)))
    print("")
    print("Smoothed test perplexity: " + str(smooth.testModel(preprocessedTest)))
    print(dashLine + "\n")

    #Change the lambda coefficients
    smooth.setLambdas(0.9, 0.1, 0)

    #Test the perplexity of the smoothed model
    print("\n" + "Smoothed Model (0.9, 0.1, 0): This may take a moment...")
    print(dashLine)
    print("Smoothed train perplexity: " + str(smooth.testModel(preprocessedTrain)))
    print("")
    print("Smoothed dev perplexity: " + str(smooth.testModel(preprocessedDev)))
    print("")
    print("Smoothed test perplexity: " + str(smooth.testModel(preprocessedTest)))
    print(dashLine + "\n")
    


def usage():
    print("A1.py accepts 1 argument:\nThe folder where the data files are stored")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        usage()
    else:
        main()

