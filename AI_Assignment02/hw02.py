from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

## Example Agent
class ReflexAgent(Agent):

  def Action(self, gameState):

    move_candidate = gameState.getLegalActions()

    scores = [self.reflex_agent_evaluationFunc(gameState, action) for action in move_candidate]
    bestScore = max(scores)
    Index = [index for index in range(len(scores)) if scores[index] == bestScore]
    get_index = random.choice(Index)

    return move_candidate[get_index]

  def reflex_agent_evaluationFunc(self, currentGameState, action):

    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    return successorGameState.getScore()



def scoreEvalFunc(currentGameState):

  return currentGameState.getScore()

class AdversialSearchAgent(Agent):

  def __init__(self, getFunc ='scoreEvalFunc', depth ='2'):
    self.index = 0
    self.evaluationFunction = util.lookup(getFunc, globals())

    self.depth = int(depth)

######################################################################################


class MinimaxAgent(AdversialSearchAgent):

  # [문제 01] MiniMax의 Action을 구현하시오. (20점)
  # (depth와 evaluation function은 위에서 정의한 self.depth and self.evaluationFunction을 사용할 것.)

  def cal_minmax(self, gameState, agentIdx, depth):

    if gameState.isLose() or gameState.isWin() or depth == self.depth:
      #print(self.evaluationFunction(gameState))
      return self.evaluationFunction(gameState)

    canMove = gameState.getLegalActions(agentIdx)

    if agentIdx == 0:
      for i in range(len(canMove)):
        if i == 0:
          max_value = -999999

        max_value = max(max_value, self.cal_minmax(gameState.generateSuccessor(agentIdx, canMove[i]), 1, depth))
      return max_value

    else:
      for j in range(len(canMove)):
        if j == 0:
          min_value = 999999

        if gameState.getNumAgents() == agentIdx+1:
          min_value = min(min_value, self.cal_minmax(gameState.generateSuccessor(agentIdx, canMove[j]), 0, depth+1))
        else:
          min_value = min(min_value, self.cal_minmax(gameState.generateSuccessor(agentIdx, canMove[j]), agentIdx+1, depth))

        #print(min_value)
      return min_value



  def Action(self, gameState):
    ####################### Write Your Code Here ################################

    canMove = gameState.getLegalActions(0)

    move = Directions.STOP
    arr = []
    maximum = -999999
    ret_idx = 0

    for i in range(len(canMove)):
      arr.append(self.cal_minmax(gameState.generateSuccessor(0, canMove[i]),1,0))
    nansoo = random.randrange(1, 11)


    if nansoo % 2 == 0:
      for i in range(len(arr)):
        if arr[i] > maximum:
          maximum = arr[i]
          ret_idx = i
    else:
      for i in range(len(arr)):
        if arr[i] >= maximum:
          maximum = arr[i]
          ret_idx = i

    ####################### initial value ##############################
    #print(maximum)
    ####################################################################

    return canMove[ret_idx]


    raise Exception("Not implemented yet")

    ############################################################################


class AlphaBetaAgent(AdversialSearchAgent):
  """
    [문제 02] AlphaBeta의 Action을 구현하시오. (25점)
    (depth와 evaluation function은 위에서 정의한 self.depth and self.evaluationFunction을 사용할 것.)
  """
  global A, B

  def cal_alphabeta(self, alpha, beta, gameState, agentIdx, depth):
    if gameState.isLose() or gameState.isWin() or depth == self.depth:
      return self.evaluationFunction(gameState)

    canMove = gameState.getLegalActions(agentIdx)

    if agentIdx == 0:
      for i in range(len(canMove)):
        if i == 0:
          max_value = -999999

        max_value = max(max_value, self.cal_alphabeta(alpha, beta, gameState.generateSuccessor(agentIdx, canMove[i]),  1, depth))

        if max_value > alpha:
          alpha = max_value

        if beta <= alpha:
          #print(beta)
          #return max_value
          break

      return max_value

    else:
      for j in range(len(canMove)):
        if j == 0:
          min_value = 999999

        if gameState.getNumAgents() == agentIdx+1:
          min_value = min(min_value, self.cal_alphabeta(alpha, beta, gameState.generateSuccessor(agentIdx, canMove[j]), 0, depth + 1))
        else:
          min_value = min(min_value, self.cal_alphabeta(alpha, beta, gameState.generateSuccessor(agentIdx, canMove[j]), agentIdx + 1, depth))

        if min_value < beta:
          beta = min_value

        if beta <= alpha:
          #print(beta)
          #return min_value
          break

      return min_value

  def Action(self, gameState):
    ####################### Write Your Code Here ################################

    A = -99999999
    B = 99999999

    canMove = gameState.getLegalActions(0)

    move = Directions.STOP
    arr = []
    maximum = -999999
    ret_idx = 0

    for i in range(len(canMove)):
      arr.append(self.cal_alphabeta(A, B, gameState.generateSuccessor(0, canMove[i]), 1, 0))

    nansoo = random.randrange(1, 11)

    if nansoo % 2 == 0:
      for i in range(len(arr)):
        if arr[i] > maximum:
          maximum = arr[i]
          ret_idx = i
    else:
      for i in range(len(arr)):
        if arr[i] >= maximum:
          maximum = arr[i]
          ret_idx = i

    return canMove[ret_idx]

    raise Exception("Not implemented yet")



    ############################################################################



class ExpectimaxAgent(AdversialSearchAgent):
  """
    [문제 03] Expectimax의 Action을 구현하시오. (25점)
    (depth와 evaluation function은 위에서 정의한 self.depth and self.evaluationFunction을 사용할 것.)
  """

  def cal_expect(self, gameState, agentIdx, depth):

    if gameState.isLose() or gameState.isWin() or depth == self.depth:
      return self.evaluationFunction(gameState)

    canMove = gameState.getLegalActions(agentIdx)

    if agentIdx == 0:
      for i in range(len(canMove)):
        if i == 0:
          max_tmp_arr = []

        max_tmp_arr.append(self.cal_expect(gameState.generateSuccessor(agentIdx, canMove[i]), 1, depth))

      max_tmp_arr.sort(reverse=True)

      max_avg = 0

      if len(max_tmp_arr) == 4:
        max_avg += 0.4 * max_tmp_arr[0] + 0.3 * max_tmp_arr[1] + 0.2 * max_tmp_arr[2] + 0.1 * max_tmp_arr[3]
      elif len(max_tmp_arr) == 3:
        max_avg += 0.5 * max_tmp_arr[0] + 0.3 * max_tmp_arr[1] + 0.2 * max_tmp_arr[2]
      elif len(max_tmp_arr) == 2:
        max_avg += 0.6 * max_tmp_arr[0] + 0.4 * max_tmp_arr[1]
      else:
        return max_tmp_arr[0]

      return max_avg

    else:
      for j in range(len(canMove)):
        if j == 0:
          min_tmp_arr = []

        if gameState.getNumAgents() == agentIdx+1:
          min_tmp_arr.append(self.cal_expect(gameState.generateSuccessor(agentIdx, canMove[j]), 0, depth + 1))
        else:
          min_tmp_arr.append(self.cal_expect(gameState.generateSuccessor(agentIdx, canMove[j]), agentIdx+1, depth))

      min_tmp_arr.sort()

      min_avg = 0

      if len(min_tmp_arr) == 4:
        min_avg += 0.4 * min_tmp_arr[0] + 0.3 * min_tmp_arr[1] + 0.2 * min_tmp_arr[2] + 0.1 * min_tmp_arr[3]
      elif len(min_tmp_arr) == 3:
        min_avg += 0.5 * min_tmp_arr[0] + 0.3 * min_tmp_arr[1] + 0.2 * min_tmp_arr[2]
      elif len(min_tmp_arr) == 2:
        min_avg += 0.6 * min_tmp_arr[0] + 0.4 * min_tmp_arr[1]
      else:
        return min_tmp_arr[0]

      return min_avg

  def Action(self, gameState):
    ####################### Write Your Code Here ################################

    canMove = gameState.getLegalActions(0)

    move = Directions.STOP
    arr = []
    maximum = -999999
    ret_idx = 0

    for i in range(len(canMove)):
      # temp = self.cal_minmax(gameState.generateSuccessor(0, canMove[i]),0,0)
      arr.append(self.cal_expect(gameState.generateSuccessor(0, canMove[i]), 1, 0))

    nansoo = random.randrange(1, 11)

    if nansoo % 2 == 0:
      for i in range(len(arr)):
        if arr[i] > maximum:
          maximum = arr[i]
          ret_idx = i
    else:
      for i in range(len(arr)):
        if arr[i] >= maximum:
          maximum = arr[i]
          ret_idx = i

    return canMove[ret_idx]


    raise Exception("Not implemented yet")

    ############################################################################
