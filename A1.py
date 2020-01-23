import sys
import re
import math
from fractions import Fraction

# Open the file in read mode
#text = open(sys.argv[1], "r")

train = "1b_benchmark.train.tokens"
dev = "1b_benchmark.dev.tokens"
test = "1b_benchmark.test.tokens"

dashLine = "------------------------------------------------------------------------"


class UnigramModel:
    def __init__(self, text):
        print("Constructing Unigram Model...")
        textFile = open(text, "r")
        self.freq_dict = dict()
        self.corpusSize = 0
        self.unkList = list()
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
        fdict = {k: v for (k, v) in idict.items() if v >= 3}
        fdict['unk'] = 0
        for k, v in idict.items():
            if(v < 3):
                self.unkList.append(k)
                fdict['unk'] = fdict['unk'] + idict[k]
        return fdict

    def calcTokenProb(self, token):
        # The numerator is the frequency of the specified token, return 0 if not found in dict
        unigram_numerator = self.freq_dict.get(token, 0)

        # If frequency is 0, the word is unkown
        if unigram_numerator == 0:
            unigram_numerator = self.freq_dict.get('unk', 0)

        # The denominator is just the size of the corpus
        unigram_denominator = self.corpusSize
        #print("Pr(" + token + "):")
        #print("numerator = ", unigram_numerator, "denominator = ", unigram_denominator)
        prob = Fraction(unigram_numerator, unigram_denominator)
        return prob

    def calcSentenceProb(self, sentence):

        # Return 0 for empty sentence
        if not sentence:
            return 0

        prob = Fraction(1)
        # The MLE for a sentence is the product of the MLEs of its constituent tokens
        for token in sentence:
            prob = prob * self.calcTokenProb(token)

        return prob

    # Takes a list of sentences and calculates the perplexity of the model given those senetences
    def calcPerplexityOLD(self, sentences):
        logSum = 0

        # Keep track of how many tokens are in the sample we are testing
        sampleSize = 0

        # Find the log probability of each sentence and sum them all up
        for sentence in sentences:
            sampleSize += len(sentence)
            sentenceProb = self.calcSentenceProb(sentence)

            # Log is undefined if input is not > 0
            if sentenceProb.numerator > 0:
                logProb = math.log(sentenceProb.numerator, 2) - \
                    math.log(sentenceProb.denominator, 2)
                logSum += logProb
            # Here we assume Pr = 0, so perplexity is infinite, return -1 to signify this
            else:
                print("Sentence:" + str(sentence))
                return -1

        # To get perplexity, multiply this sum by the negative reciprocal
        # of sample size and exponentiate it base 2
        logSum = logSum * (-1 / float(sampleSize))
        perplexity = 2 ** logSum

        return perplexity

    def calcPerplexity(self, sentences):
        sampleLogSum = 0

        # Keep track of how many tokens are in the sample we are testing
        sampleSize = 0

        # Find the log probability of each sentence and sum them all up
        for sentence in sentences:
            sampleSize += len(sentence)

            sentenceLogSum = 0

            # The log probability of each sentence is the sum of the log probabilities
            # of its constituent tokens
            for token in sentence:
                tokenProb = self.calcTokenProb(token)

                if tokenProb > 0:
                    sentenceLogSum += math.log(tokenProb.numerator, 2) - \
                        math.log(tokenProb.denominator, 2)
                else:
                    print("Token: " + token)
                    return -1

            sampleLogSum += sentenceLogSum

        # To get perplexity, multiply this sum by the negative reciprocal
        # of sample size and exponentiate it base 2
        sampleLogSum = sampleLogSum * (-1 / float(sampleSize))
        perplexity = 2 ** sampleLogSum

        return perplexity

    # Uses a test file and calculates perplexity based on that sample

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

# textFile = open(text, "r")
#         self.freq_dict = dict()
#         self.corpusSize = 0
#         for line in textFile:
#             # append stop token to EOLs
#             line = line + ' <STOP>'
#             # split the line into tokens
#             tokens = line.split(" ")
#             # iterate through each token
#             for token in tokens:
#                 self.corpusSize += 1
#                 # remove whitespace and \n characters
#                 token = token.strip()
#                 # if token is already in dictionary then increment the count
#                 if token in self.freq_dict:
#                     self.freq_dict[token] = self.freq_dict[token] + 1
#                 # else initialize that token's key with value 1
#                 else:
#                     self.freq_dict[token] = 1

def main():

    path = sys.argv[1] + "/"

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


if __name__ == '__main__':
    main()
