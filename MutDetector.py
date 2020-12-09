import re
from os import listdir
from os.path import basename

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import SeqIO
from scipy.stats import chisquare


baseDict = {
    'A' : 0,
    'T' : 1,
    'C' : 2,
    'G' : 3
}

baseDict_rev = {
    0 : 'A',
    1 : 'T',
    2 : 'C',
    3 : 'G'
}

def pileupBaseCounter(resultStr):
    # Count base calls numbers from a pileup result string
    
    counts = np.zeros(4, dtype=np.int32)
    i = 0
    while i < len(resultStr):
        baseSign = resultStr[i].upper()
        if baseSign in ['A', 'T', 'C', 'G']:
            counts[baseDict[baseSign]] += 1
        elif baseSign in ['+', '-']:
            j = i + 1
            while resultStr[j].isdigit():
                j += 1
            j -= 1
            skipN = int(resultStr[i + 1: j + 1])
#             print(i, j, skipN)
            i = j + skipN + 1
        elif baseSign == '^':
            i += 1
#         else: 
#             print(baseSign, end=' ')
        i += 1
    return counts

def pileupParser(filePath):
    # Read pileup file and add a column ('Calls') that sumerize the number of each base
    
    with open(filePath) as f:
        file = pd.read_csv(f, sep='\t', names=['Chr', 'Pos', 'RefBase', 'Depth', 'Result', 'Qual'], index_col=[0, 1])
    file['Calls'] = file['Result'].apply(pileupBaseCounter)

    return file

def getCallArr(pileupPath, Chr, refLen):
    # Read pileup file and generate the base-calls of a certain chromosome to a numpy array
    
    pileup = pileupParser(pileupPath)
    callArr = np.zeros([refLen, 4])

    callDict = dict(pileup.loc[Chr]['Calls'])
    for pos in callDict:
        callArr[pos-1] = callDict[pos] 

    return callArr

def getConvArr(callArr, ref):
    # Calculate the over all sequencing convertion error rate
    # print(callArr.shape)
    # print(len(ref))
    conversionArr = np.zeros((4, 4))
    
    for i in range(len(ref)):
#         print(callArr[i])
        conversionArr[baseDict[ref[i]]] += callArr[i]

    conversionArr = conversionArr / conversionArr.sum(1)
    return conversionArr

def getExpFr(Arr, convArr):
    # Get the expexted seq-calls from the reference call array and a conversion matrix
    
    psudoFr = Arr + 1
    convFr = psudoFr @ convArr
    normFr = (convFr.T / convFr.sum(1)).T
    
    return normFr   



class MutDetector():
    baseDict = baseDict

    def __init__(self, genomeRef, parental, knownSeq=None):
        # genomeRef: the referance fasta file
        # parental: the pileup file of the parental line
        # knownSeq: sequencing results of a know sequence, that's for the sequencing error rate.
        
        if knownSeq is None:
            knownSeq = parental

        with open(genomeRef, 'r') as f:
            gRef_fa = list(SeqIO.parse(f, 'fasta'))

        self.chrID = gRef_fa[0].id
        self.refSeq = gRef_fa[0].upper()
        
        self.knownSeqArr = getCallArr(knownSeq, gRef_fa[0].id, len(self.refSeq))
        self.parentalArr = getCallArr(parental, gRef_fa[0].id, len(self.refSeq))

        self.convArr = getConvArr(self.knownSeqArr, self.refSeq)

        self.expFr = getExpFr(self.parentalArr, self.convArr)

        print('Building of mutation detector completed. \
            \n{0} is used as reference \
            \n{1} is used for estimating sequencing error \
            \n{2} is used as the non-mutated strain'.format(genomeRef, knownSeq, parental))

    def callSNP(self, pileupPath):
        calls =  getCallArr(pileupPath, self.chrID, len(self.refSeq))
        expCalls =  (self.expFr.T * calls.sum(1)).T
        
        chisquares = chisquare(calls.T, expCalls.T)
        
        return calls, expCalls, chisquares.pvalue

    def callSgnfctSNP(self, filePath, pThreshold=0.005, refRange=None):
        # Bonferroni correction included for the p-values

        calls, expCalls, pValues = self.callSNP(filePath)
        pValues *= len(self.refSeq)
        sortedIdx = pValues.argsort()
        sortedp = pValues[sortedIdx]

        sigs = sortedp < pThreshold
        if not (refRange is None):
            sigs = np.logical_and(sigs, np.logical_and(refRange[0] < sortedIdx, sortedIdx < refRange[1]))

        outputList = [calls, [], []]

        print(basename(filePath)[0:-7], '{0} hits found'.format(sigs.sum()))
        print('Printing in format: Index(zero based) \t Bonferroni p-value \t Call counts (A-T-C-G) \t Expected counts (A-T-C-G)')

        with np.printoptions(precision=1, suppress=True):
            for idx in sortedIdx[sigs]:
                print(idx, '{:.2e}'.format(pValues[idx]), calls[idx], expCalls[idx], sep='\t')
                outputList[1].append(idx)
                outputList[2].append(pValues[idx])
        print()

        return basename(filePath)[0:-7], outputList

