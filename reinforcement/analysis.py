# analysis.py
# -----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2a():
    """
      Prefer the close exit (+1), risking the cliff (-10).
      
      To risk the cliff and prefer close exit:
      - Low discount: don't care much about future, so +10 isn't worth the extra steps
      - Low/zero noise: can safely risk the cliff without accidentally falling
      - Small negative or zero living reward: don't penalize taking steps much
    """
    answerDiscount = 0.1
    answerNoise = 0.0
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward

def question2b():
    """
      Prefer the close exit (+1), but avoiding the cliff (-10).
      
      To avoid the cliff but still prefer close exit:
      - Low discount: don't care much about future, so +10 isn't worth the extra steps
      - Higher noise: risking the cliff is too dangerous, might fall
      - Small negative living reward: penalty for taking too many steps keeps us from going far
    """
    answerDiscount = 0.2
    answerNoise = 0.2
    answerLivingReward = -0.5
    return answerDiscount, answerNoise, answerLivingReward

def question2c():
    """
      Prefer the distant exit (+10), risking the cliff (-10).
      
      To risk the cliff and prefer distant exit:
      - High discount: care about future rewards, so +10 is worth it
      - Low/zero noise: can safely risk the cliff without accidentally falling
      - Small negative or zero living reward: don't penalize taking steps much
    """
    answerDiscount = 0.9
    answerNoise = 0.0
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward

def question2d():
    """
      Prefer the distant exit (+10), avoiding the cliff (-10).
      
      To avoid the cliff and prefer distant exit:
      - High discount: care about future rewards, so +10 is worth it
      - Higher noise: risking the cliff is too dangerous, might fall
      - Small negative or zero living reward: don't penalize taking steps much
    """
    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward

def question2e():
    """
      Avoid both exits and the cliff (so an episode should never terminate).
      
      To avoid terminating:
      - High positive living reward: make it better to keep living than to exit
      - The living reward should be high enough that even +10 exit isn't worth it
    """
    answerDiscount = 0.9
    answerNoise = 0.0
    answerLivingReward = 1.0
    return answerDiscount, answerNoise, answerLivingReward

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))