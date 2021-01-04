# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util ,math

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        #newGhostStates = successorGameState.getGhostStates()
        #newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = 0.0
        base = 100.0

        #if get fodd then add score
        score += currentGameState.hasFood(newPos[0], newPos[1]) * base
        for food in newFood.asList():
            score -= base * (1 - math.exp(-1.0 * util.manhattanDistance(newPos, food)))
            
        #if too close to ghost then minus points
        GhostPos = successorGameState.getGhostState(1).getPosition()
        GhostScareTime = successorGameState.getGhostState(1).scaredTimer
        if util.manhattanDistance(newPos, GhostPos) < 2 and GhostScareTime ==0:
            score -= 1e100

        #Stop is not good for win so minus points
        if action == Directions.STOP:
            score -= base

        return score
        #return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """
    
    def minimax(self, state, depth, agent = 0, MaxStep = True):
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state), Directions.STOP
        actions = state.getLegalActions(agent)
        if MaxStep:
            scores=[]
            for action in actions:
                scores.append(self.minimax(state.generateSuccessor(agent, action), depth - 1, 1, False)[0])
            MaxScore = max(scores)
            Dir = []
            for i in range(len(scores)):
                if scores[i]==MaxScore:
                    Dir.append(i)
            return MaxScore, actions[random.choice(Dir)]
        else:
            scores = []
            
            for action in actions:
                if agent == state.getNumAgents() - 1:#if last ghost then return to max
                    scores.append(self.minimax(state.generateSuccessor(agent, action), depth - 1, 0, True)[0])
                else : #otherwise we have to compute other ghosts
                    scores.append(self.minimax(state.generateSuccessor(agent, action), depth , agent+1, False)[0])
            MinScore = min(scores)
            Dir = []
            for i in range(len(scores)):
                if scores[i]==MinScore:
                    Dir.append(i)
            return MinScore, actions[random.choice(Dir)]

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        #scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        #bestScore = max(scores)
        #bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        #chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        return self.minimax(gameState, self.depth * 2, 0, True)[1]
        util.raiseNotDefined()
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def alphabetapruning(self, state, depth,alpha,beta, agent = 0, MaxStep = True):
        #if died or find to the root then stop
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state), Directions.STOP
        actions = state.getLegalActions(agent)
        if MaxStep:
            MaxScore = float("-inf")
            MaxAction =[]
            for action in actions:
                score = (self.alphabetapruning(state.generateSuccessor(agent, action), depth - 1,alpha,beta, 1, False)[0])
                alpha = max(score,alpha)
                if score > MaxScore:
                    MaxScore = score
                    MaxAction = [action]
                elif score == MaxScore:
                    MaxAction.append(action)
                if MaxScore > beta:
                    break;
            return MaxScore, random.choice(MaxAction)
        else:
            MinScore = float("inf")
            MinAction =[]
            for action in actions:
                if agent == state.getNumAgents() - 1:#if last ghost then return to max
                    score = (self.alphabetapruning(state.generateSuccessor(agent, action), depth - 1,alpha,beta, 0, True)[0])
                    beta = min(score,beta)
                    if score < MinScore:
                        MinScore = score
                        MinAction = [action]
                    elif score == MinScore:
                        MinAction.append(action)
                    if MinScore < alpha:
                        break;
                else : #otherwise we have to compute other ghosts
                    score = (self.alphabetapruning(state.generateSuccessor(agent, action), depth,alpha,beta, agent+1, False)[0])
                    beta = min(score,beta)
                    if score < MinScore:
                        MinScore = score
                        MinAction = [action]
                    elif score == MinScore:
                        MinAction.append(action)
                    if MinScore < alpha:
                        break;
            return MinScore, random.choice(MinAction)
    
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.alphabetapruning(gameState, self.depth * 2,float("-inf"),float("inf"), 0, True)[1]
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.expectimax(gameState,self.depth * 2,0,True)[1]
        util.raiseNotDefined()
    def expectimax(self, state, depth, agent = 0, MaxStep = True):
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state), Directions.STOP
        actions = state.getLegalActions(agent)
        if MaxStep:
            scores=[]
            for action in actions:
                scores.append(self.expectimax(state.generateSuccessor(agent, action), depth - 1, 1, False)[0])
            MaxScore = max(scores)
            Dir = []
            for i in range(len(scores)):
                if scores[i]==MaxScore:
                    Dir.append(i)
            return MaxScore, actions[random.choice(Dir)]
        else:
            scores = []
            for action in actions:
                if agent == state.getNumAgents() - 1:#if last ghost then return to max
                    scores.append(self.expectimax(state.generateSuccessor(agent, action), depth - 1, 0, True)[0])
                else : #otherwise we have to compute other ghosts
                    scores.append(self.expectimax(state.generateSuccessor(agent, action), depth , agent+1, False)[0])
            #here is the difference that we assume the enemy might choose suboptimal solution so that we return the average max
            return float(sum(scores)) / float(len(scores)), None
        
    

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
      I think the location of capsule should also be considered 
      because it would increase the scaredtime of ghost
    """
    "*** YOUR CODE HERE ***"
    baseScores = [50, 300] #capsule is more important than food
    pos = currentGameState.getPacmanPosition()

    score = currentGameState.getScore() - 20 * currentGameState.getNumFood()

    foodList = currentGameState.getFood().asList()
    #when food is far away then minus more points
    for food in foodList:
        score -= baseScores[0] * (1 - math.exp(-1.0  * util.manhattanDistance(pos, food)))

    capsuleList = currentGameState.data.capsules
    #when capsule is far away then minus more points
    for capsule in capsuleList:
        score -= baseScores[1] * (1 - math.exp(-1.0  * util.manhattanDistance(pos, capsule)))

    ghostList = currentGameState.getGhostPositions()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    #if ghost is closed and not scared then minus points
    for ghost in zip(ghostList, newScaredTimes):
        if util.manhattanDistance(pos, ghost[0]) < 2 and ghost[1]==0:
            score -= 1e10
    return score
    
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

