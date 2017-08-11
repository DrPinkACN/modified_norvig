import re, collections
import numpy as np
import os, sys
from scipy.sparse import dok_matrix
from itertools import product

class spelcor(object):

    def __init__(self, alphabet='[a-zA-Z]+', wordsfile=None, freqfile=False, lower=False, delimiter=None):
        """
        `alphabet` is a simple string containing all characters included in
        regex find all expression.
        RegEx ranges, like `a-z` will work, a hyphon will be added to the end of the
        alphabet if one doesn't exist to treat hyphonated entries in words list as a
        single word.
        `words` is a text file containing pieces of writing you want to spell correct against.
        If one is not supplied, the big text will be used.
        based on: Spelling correction from [Peter Norvig](http://norvig.com/spell-correct.html):
        """
        #srcdir = os.path.dirname(__file__)

        try:
            if alphabet[0] != "[" and alphabet[-2:] != "]+":
                raise Exception
        except:
            print("""ERROR: your alphabet must be in the form of a RegEx findall expression.
            this should include square brackets and a trailing '+' symbol, e.g., '[a-z]+' or '[abcABC&]+'""")

        self.alphabet = ''.join(re.findall(alphabet,
                                           '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,./:;<=>?@[]^_`{|}-'))
        self.alphabet = ''.join([i if (ord(i) < 128 and ord(i) != 32) else '' for i in self.alphabet])
        if self.alphabet[-1] != '-':
            self.alphabet += '-'
        self.lower = lower

        # load words
        if not (wordsfile):
            with open(os.path.join(srcdir,r'big.txt'),'r') as fn:
                text=fn.read()
        elif freqfile:
            text=collections.defaultdict(int)
            with open(wordsfile, 'r') as fn:
                for i in fn.readlines():
                    entry=i.strip().split(delimiter)
                    if lower:text[entry[0].lower()]+=int(entry[1])
                    else: text[entry[0]]+=int(entry[1])
        else:
            with open(wordsfile, 'r') as fn:
                text = fn.read()
        # create a word list
        if freqfile:
            self.model = text
        elif self.lower:
            words = re.findall('[' + self.alphabet + ']+', text.lower())
            # make the model
            self.model = collections.defaultdict(int)
            for w in words:
                self.model[w] += 1
        else:
            words = re.findall('[' + self.alphabet + ']+', text)
            # make the model
            self.model = collections.defaultdict(int)
            for w in words:
                self.model[w] += 1
                if w.islower():
                    self.model[w.capitalize()] += 1
        self.model=dict(self.model)
        self.unshifted = """1234567890-=
        qwertyuiop[]
        asdfghjkl;'
        zxcvbnm,./"""
        self.shifted = """!@#$%^&*()_+
        QWERTYUIOP{}|
        ASDFGHJKL:\"
        ZXCVBNM<>?"""

        self.unshifted = self.unshifted.replace(" ", "")
        self.shifted = self.shifted.replace(" ", "")

        self.xdim = np.max([len(i) for i in (self.unshifted + '\n' + self.shifted).split('\n')])
        self.ydim = len(self.shifted.split('\n'))
        self.zdim = 2
        self.kb = np.zeros((self.xdim, self.ydim, self.zdim), dtype='str')

        for i in range(len(self.unshifted.split('\n'))):
            for ii in range(len(self.unshifted.split('\n')[i])):
                self.kb[ii, i, 0] = self.unshifted.split('\n')[i][ii]
        for i in range(len(self.shifted.split('\n'))):
            for ii in range(len(self.shifted.split('\n')[i])):
                self.kb[ii, i, 1] = self.shifted.split('\n')[i][ii]

    def kbmatch(self, word1, word2):
        word1 = word1.replace(" ", "")
        word2 = word2.replace(" ", "")
        word1 = set([i for i in word1])
        word2 = set([i for i in word2])
        # get the letters locations for all letters in word1
        word1ls = np.unique(np.array(list(word1)))
        word2ls = np.unique(np.array(list(word2)))
        kbw1 = np.zeros((self.xdim, self.ydim, self.zdim))
        kbw2 = np.zeros((self.xdim, self.ydim, self.zdim))

        for i in word1ls: kbw1 += (self.kb == i)
        kbw1 = kbw1 > 0
        w1locs = np.where(kbw1)

        for i in word2ls: kbw2 += (self.kb == i)
        kbw2 = kbw2 > 0
        w2locs = np.where(kbw2)

        # get all letters locations for letters in word2 that diffe
        worddif21 = (word2 - word1)
        worddif12 = (word1 - word2)
        kbwd21 = np.zeros((self.xdim, self.ydim, self.zdim))
        kbwd12 = np.zeros((self.xdim, self.ydim, self.zdim))
        for i in np.array(list(worddif21)): kbwd21 += (self.kb == i)
        for i in np.array(list(worddif12)): kbwd12 += (self.kb == i)

        kbwd12 = kbwd12 > 0
        kbwd21 = kbwd21 > 0

        wdlocs12 = np.where(kbwd12)
        wdlocs21 = np.where(kbwd21)

        totdist = []

        for i in np.array(wdlocs21).T:
            dists = []
            for ii in np.array(w1locs).T:
                space=i-ii
                space[2]=space[2]/2. #shited letters are 'close'
                dists.append(np.sqrt(np.sum(space**2.)))
            totdist.append(np.exp(-1 * np.min(dists)))

        for i in np.array(wdlocs12).T:
            dists = []
            for ii in np.array(w2locs).T:
                space=i-ii
                space[2]=space[2]/2. #shited letters are 'close'
                dists.append(np.sqrt(np.sum(space**2.)))
            totdist.append(np.exp(-1 * np.min(dists)))

        if len(totdist) == 0:
            return 1.0
        else:
            return np.sum(totdist) / (len(totdist) ** 2.)

    def _perms(self, word):
        deletes=[]
        transposes=[]
        replaces=[]
        inserts=[]
        words=[word] #this list can be expanded to include other word modifications
        for w in words:
            splits = [(w[:i], w[i:]) for i in range(len(w) + 1)]
            deletes += [a + b[1:] for a, b in splits if b]
            transposes += [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
            replaces += [a + c + b[1:] for a, b in splits for c in self.alphabet if b]
            inserts += [a + c + b for a, b in splits for c in self.alphabet]
        return set(deletes + transposes + replaces + inserts)

    def _perms2(self, word):
        return set(e2 for e1 in self._perms(word) for e2 in self._perms(e1) if e2 in self.model)

    def _kperms(self, words):
        return set(w for w in words if w in self.model)

    def cor_norvig(self, word):
        if self.lower: word.lower()
        self.candidates = self._kperms([word]) or self._kperms(self._perms(word)) or self._perms2(word) or [word]
        return max(self.candidates, key=self.model.get)

    def cor_modNorvig(self, word):
        try:
            return self.recs(word)[0][0]
        except:
            return word

    def recs(self, word):
        if self.lower: word.lower()
        self.candidates = self._kperms([word]) or self._kperms(self._perms(word)) or self._perms2(word) or [word]
        self.kbmodel = []
        for i in self.candidates:
            if i in self.model:
                self.kbmodel.append([i, self.kbmatch(i, word), self.model[i]])
        self.candidates=sorted(self.kbmodel, key=lambda x: (x[1], x[2]))[::-1] #to store the sorted, correct candidates
        return [(i[0],i[1]) for i in self.candidates]
