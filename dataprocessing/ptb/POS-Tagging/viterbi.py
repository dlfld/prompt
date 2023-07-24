
'''
Yihui Peng
Ze Xuan Ong
Jocelyn Huang
Noah A. Smith
Yifan Xu

Usage: python viterbi.py <HMM_FILE> <TEXT_FILE> <OUTPUT_FILE>

Apart from writing the output to a file, the program also prints
the number of text lines read and processed, and the time taken
for the entire program to run in seconds. This may be useful to
let you know how much time you have to get a coffee in subsequent
iterations.
'''

import math
import sys
import time

from collections import defaultdict

# Magic strings and numbers
HMM_FILE = sys.argv[1]
TEXT_FILE = sys.argv[2]
OUTPUT_FILE = sys.argv[3]
TRANSITION_TAG = "trans"
EMISSION_TAG = "emit"
OOV_WORD = "OOV"         # check that the HMM file uses this same string
INIT_STATE = "init"      # check that the HMM file uses this same string
FINAL_STATE = "final"    # check that the HMM file uses this same string


class Viterbi():
    def __init__(self):
        # transition and emission probabilities. Remember that we're not dealing with smoothing 
        # here. So for the probability of transition and emission of tokens/tags that we haven't 
        # seen in the training set, we ignore thm by setting the probability an impossible value 
        # of 1.0 (1.0 is impossible because we're in log space)

        self.transition = defaultdict(lambda: defaultdict(lambda: 1.0))
        self.emission = defaultdict(lambda: defaultdict(lambda: 1.0))
        # keep track of states to iterate over 
        self.states = set()
        self.POSStates = set()
        # store vocab to check for OOV words
        self.vocab = set()

        # text to run viterbi with
        self.text_file_lines = []
        with open(TEXT_FILE, "r") as f:
            self.text_file_lines = f.readlines()

    def readModel(self):
        # Read HMM transition and emission probabilities
        # Probabilities are converted into LOG SPACE!
        with open(HMM_FILE, "r") as f:
            for line in f:
                line = line.split()

                # Read transition
                # Example line: trans NN NNPS 9.026968067100463e-05
                # Read in states as prev_state -> state
                if line[0] == TRANSITION_TAG:
                    (prev_state, state, trans_prob) = line[1:4]
                    self.transition[prev_state][state] = math.log(float(trans_prob))
                    self.states.add(prev_state)
                    self.states.add(state)

                # Read in states as state -> word
                elif line[0] == EMISSION_TAG:
                    (state, word, emit_prob) = line[1:4]
                    self.emission[state][word] = math.log(float(emit_prob))
                    self.states.add(state)
                    self.vocab.add(word)

        # Keep track of the non-initial and non-final states
        self.POSStates = self.states.copy()
        self.POSStates.remove(INIT_STATE)
        self.POSStates.remove(FINAL_STATE)

    # run Viterbi algorithm and write the output to the output file
    def runViterbi(self):
        result = []
        for line in self.text_file_lines:
            result.append(self.viterbiLine(line))

        # Print output to file
        with open(OUTPUT_FILE, "w") as f:
            for line in result:
                f.write(line)
                f.write("\n")
        return result

    #main func
    def viterbiLine(self, line):
        words = line.split()
        self.V = defaultdict(lambda: defaultdict(lambda: 1.0))
        self.BP = defaultdict(lambda: defaultdict(lambda: 1.0))
        result = []

        states_list = list(self.states)

        for i in range(len(states_list)):
            if words[0] not in self.vocab:
                words[0] = OOV_WORD
  
            a = self.transition[INIT_STATE][states_list[i]]
            b = self.emission[states_list[i]][words[0]]

            if a <= 0:
                if b <= 0:
                    self.V[states_list[i]][0] = a + b
                    self.BP[states_list[i]][0] = states_list[i]

        for (i, word) in enumerate(words):
            if word not in self.vocab:
                word = OOV_WORD

            if i != 0:
                for j in range(len(states_list)):
                    cur_max_state = ""
                    cur_max = -math.inf
                    for k in range(len(states_list)):
                        val = (self.V[states_list[k]][i-1] 
                            + self.transition[states_list[k]][states_list[j]] 
                            + self.emission[states_list[j]][word])
                        if (self.transition[states_list[k]][states_list[j]] <= 0 
                            and self.emission[states_list[j]][word] <= 0 
                            and self.V[states_list[k]][i-1] != 1.0 and val > cur_max):

                            cur_max_state = states_list[k]
                            cur_max = val

                    if cur_max != -math.inf:
                        self.V[states_list[j]][i] = cur_max
                        self.BP[states_list[j]][i] =  cur_max_state

        for i in range(len(states_list)):
            if (self.transition[states_list[i]][FINAL_STATE] <= 0 
                and self.V[states_list[i]][len(words)-1] <= 0):

                c = self.transition[states_list[i]][FINAL_STATE]
                d = self.V[states_list[i]][len(words)-1]

                self.V[states_list[i]][len(words)] = c + d

        cur_max_state = ""       
        cur_max = -math.inf

        for i in range(len(states_list)):
            if self.V[states_list[i]][len(words)] != 1.0 and cur_max < self.V[states_list[i]][len(words)]:
                cur_max_state = states_list[i]
                cur_max = self.V[states_list[i]][len(words)]
            elif self.V[states_list[i]][len(words)] != 1.0 and cur_max >= self.V[states_list[i]][len(words)]:
                    return ""

        result = [cur_max_state]

        for i in reversed(range(1, len(words))):
            if self.BP[cur_max_state][i] != 1.0:
                result.append(self.BP[cur_max_state][i])  
                cur_max_state = self.BP[cur_max_state][i]     
        
        new_res = " ".join(result[::-1])
        return new_res

if __name__ == "__main__":
    # Mark start time
    t0 = time.time()
    viterbi = Viterbi()
    viterbi.readModel()
    viterbi.runViterbi()
    # Mark end time
    t1 = time.time()
    print("Time taken to run: {}".format(t1 - t0))

