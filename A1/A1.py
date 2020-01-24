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

    print("Preprocessing train data...")

    trainFile = open(trainPath, "r")

    freq_dict = dict()
    corpusSize = 0

    #Go through the train file and build a vocabulary 
    for line in trainFile:

        line = '<START> ' + line + ' <STOP>'
        tokens = line.split(" ")
        corpusSize += len(tokens) - 1

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

    #Go through all three files and unk any words that need to be
    #Write these to new files
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

    print("Train data preprocessed!")
    print("Preprocessing dev data...")

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

    print("Dev data preprocessed!")
    print("Preprocessing test data...")

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

    print("Test data preprocessed!")

    return (vocabulary, corpusSize)

class UnigramModel:
    def __init__(self, vocab, corpusSize):
        print("Constructing Unigram Model...")

        self.unigramFrequency = vocab
        self.corpusSize = corpusSize

        print("Unigram model constructed!")

    def calcTokenProb(self, token):
        #The numerator is the frequency of the specified token, return 0 if not found in dict
        unigram_numerator = self.unigramFrequency.get(token, 0)

        #If frequency is 0, the word is unkown
        if unigram_numerator == 0:
            unigram_numerator = self.unigramFrequency.get('<UNK>', 0)

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
            line = line + ' <STOP>'

            # split the line into tokens and strip whitespace
            sentence = [token.strip() for token in line.split(" ")]

            sentences.append(sentence)

        testFile.close()

        return self.calcPerplexity(sentences)


class BigramModel:
    def __init__(self, text, unigramModel):
        self.uni = unigramModel
        textFile = open(text, "r")
        self.freq_dict = dict()
        self.corpusSize = 0
        for line in textFile:
             # append stop token to EOLs
            line = '<START>' + line + ' <STOP>'
             # split the line into tokens
            tokens = line.split(" ")
            bigramList = self.createBigrams(tokens)
            for bigram in bigramList:
                 if bigram in self.freq_dict:
                     self.freq_dict[bigram] = self.freq_dict[bigram] + 1
                 # else initialize that token's key with value 1
                 else:
                     self.freq_dict[bigram] = 1
        self.replaceUNKs(self.freq_dict, self.uni.unkList)

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
            bigramList.append((sentence[i - 1].strip(), sentence[i].strip()))
            i += 1

        return bigramList


def main():

    path = sys.argv[1] + "/"

    print("\n" + dashLine)
    (vocabulary, corpusSize) = preprocessData(path + train, path + dev, path + test)
    print(dashLine + "\n")

    print("\n" + dashLine)
    uni = NewUnigram(vocabulary, corpusSize)
    print(dashLine + "\n")

    print("\n" + dashLine)
    print("Unigram train perplexity: " + str(uni.testModel(path + train)))
    print("")
    print("Unigram dev perplexity: " + str(uni.testModel(path + dev)))
    print("")
    print("Unigram test perplexity: " + str(uni.testModel(path + test)))
    


def usage():
    print("A1.py accepts 1 argument:\nThe folder where the data files are stored")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        usage()
    else:
        main()

