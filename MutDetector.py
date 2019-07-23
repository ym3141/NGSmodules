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

def pileupBaseCounter(resultStr):
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
    with open(filePath) as f:
        file = pd.read_csv(f, sep='\t', names=['Chr', 'Pos', 'RefBase', 'Depth', 'Result', 'Qual'], index_col=[0, 1])
    file['Calls'] = file['Result'].apply(pileupBaseCounter)
    return file

def callArr(pileupPath, Chr):
    pileup = pileupParser(pileupPath)
    callArr = np.array(list(pileup.loc[Chr]['Calls']))
    return callArr

def getConvArr(callArr, ref):
    conversionArr = np.zeros((4, 4))
    for i in range(len(ref)):
#         print(callArr[i])
        conversionArr[baseDict[ref[i]]] += callArr[i]
    conversionArr = conversionArr / conversionArr.sum(1)
    return conversionArr

def getExpFr(Arr, convArr):
    psudoFr = Arr + 1
    convFr = psudoFr @ convArr
    normFr = (convFr.T / convFr.sum(1)).T
    
    return normFr   



class MutDetector():
    baseDict = baseDict

    def __init__(self, genomeRef, parental, knownSeq=None):
        if knownSeq is None:
            knownSeq = parental

        with open(genomeRef, 'r') as f:
            gRef_fa = list(SeqIO.parse(f, 'fasta'))

        self.chrID = gRef_fa[0].id
        self.refSeq = gRef_fa[0].upper()
        
        self.knownSeqArr = callArr(knownSeq, gRef_fa[0].id)
        self.parentalArr = callArr(parental, gRef_fa[0].id)

        self.convArr = getConvArr(self.knownSeqArr, self.refSeq)

        self.expFr = getExpFr(self.parentalArr, self.convArr)

    def callSNP(self, pileupPath):
        calls =  callArr(pileupPath, self.chrID)
        expCalls =  (self.expFr.T * calls.sum(1)).T
        
        pValues = chisquare(calls.T, expCalls.T).pvalue
        
        return calls, expCalls, pValues

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
        for idx in sortedIdx[sigs]:
            print(idx, '{:.2e}'.format(pValues[idx]), calls[idx], ['{:.1f}'.format(i) for i in expCalls[idx]], sep='\t')
            outputList[1].append(idx)
            outputList[2].append(pValues[idx])

        return basename(filePath)[0:-7], outputList

