from pyMaze import maze,agent,COLOR,textLabel
from queue import PriorityQueue
import numpy as np
from timeit import timeit
import timeit
import sys
import gc
from collections import deque

def DFS():
    m=maze()
    # m.CreateMaze(loadMaze='maze--2023-03-10--13-30-20.csv')  # 5 X 5 Maze
    # m.CreateMaze(loadMaze='maze--2023-03-10--13-30-36.csv')  # 4 X 7 Maze
    m.CreateMaze(loadMaze='maze--2023-03-07--11-07-26.csv')  # 10 X 10 Maze

    start_time = timeit.default_timer()

    start=(m.rows,m.cols)
    explored=[start]
    frontier=[start]
    dfsPath={}
    while len(frontier)>0:
        currCell=frontier.pop()
        if currCell==(1,1):
            break
        for d in 'ESNW':
            if m.maze_map[currCell][d]==True:
                if d=='E':
                    childCell=(currCell[0],currCell[1]+1)
                elif d=='W':
                    childCell=(currCell[0],currCell[1]-1)
                elif d=='S':
                    childCell=(currCell[0]+1,currCell[1])
                elif d=='N':
                    childCell=(currCell[0]-1,currCell[1])
                if childCell in explored:
                    continue
                explored.append(childCell)
                frontier.append(childCell)
                dfsPath[childCell]=currCell
    fwdPath={}
    cell=(1,1)
    while cell!=start:
        fwdPath[dfsPath[cell]]=cell
        cell=dfsPath[cell]
    end_time = timeit.default_timer()
    memory_used = memory_usage()
    time_taken = end_time - start_time

    print("Memory used by my_function: {} bytes".format(memory_used))
    print("Execution time for my_function: {} seconds".format(time_taken))
    a=agent(m,footprints=True,filled=True)
    m.tracePath({a:fwdPath},delay=100)
    l=textLabel(m,'Length of Shortest Path',len(fwdPath)+1)
    m.run()
    

def DFSTest(m,start=None):
    if start is None:
        start=(m.rows,m.cols)
    explored=[start]
    frontier=[start]
    dfsPath={}
    dSeacrh=[]
    while len(frontier)>0:
        currCell=frontier.pop()
        dSeacrh.append(currCell)
        if currCell==m._goal:
            break
        poss=0
        for d in 'ESNW':
            if m.maze_map[currCell][d]==True:
                if d =='E':
                    child=(currCell[0],currCell[1]+1)
                if d =='W':
                    child=(currCell[0],currCell[1]-1)
                if d =='N':
                    child=(currCell[0]-1,currCell[1])
                if d =='S':
                    child=(currCell[0]+1,currCell[1])
                if child in explored:
                    continue
                poss+=1
                explored.append(child)
                frontier.append(child)
                dfsPath[child]=currCell
        if poss>1:
            m.markCells.append(currCell)
    fwdPath={}
    cell=m._goal
    while cell!=start:
        fwdPath[dfsPath[cell]]=cell
        cell=dfsPath[cell]
    return dSeacrh,dfsPath,fwdPath

def BFS():
    m=maze()
    # m.CreateMaze(loadMaze='maze--2023-03-10--13-30-20.csv')  # 5 X 5 Maze
    # m.CreateMaze(loadMaze='maze--2023-03-10--13-30-36.csv')  # 4 X 7 Maze
    m.CreateMaze(loadMaze='maze--2023-03-07--11-07-26.csv')  # 10 X 10 Maze

    start_time = timeit.default_timer()
    start=(m.rows,m.cols)
    frontier=[start]
    explored=[start]
    bfsPath={}
    while len(frontier)>0:
        currCell=frontier.pop(0)
        if currCell==(1,1):
            break
        for d in 'ESNW':
            if m.maze_map[currCell][d]==True:
                if d=='E':
                    childCell=(currCell[0],currCell[1]+1)
                elif d=='W':
                    childCell=(currCell[0],currCell[1]-1)
                elif d=='N':
                    childCell=(currCell[0]-1,currCell[1])
                elif d=='S':
                    childCell=(currCell[0]+1,currCell[1])
                if childCell in explored:
                    continue
                frontier.append(childCell)
                explored.append(childCell)
                bfsPath[childCell]=currCell
    fwdPath={}
    cell=(1,1)
    while cell!=start:
        fwdPath[bfsPath[cell]]=cell
        cell=bfsPath[cell]
    end_time = timeit.default_timer()
    memory_used = memory_usage()
    time_taken = end_time - start_time

    print("Memory used by my_function: {} bytes".format(memory_used))
    print("Execution time for my_function: {} seconds".format(time_taken))
    a=agent(m,footprints=True,filled=True)
    m.tracePath({a:fwdPath},delay=100)
    l=textLabel(m,'Length of Shortest Path',len(fwdPath)+1)
    m.run()    
    return fwdPath

def BFSTest(m,start=None):
    if start is None:
        start=(m.rows,m.cols)
    frontier = deque()
    frontier.append(start)
    bfsPath = {}
    explored = [start]
    bSearch=[]

    while len(frontier)>0:
        currCell=frontier.popleft()
        if currCell==m._goal:
            break
        for d in 'ESNW':
            if m.maze_map[currCell][d]==True:
                if d=='E':
                    childCell=(currCell[0],currCell[1]+1)
                elif d=='W':
                    childCell=(currCell[0],currCell[1]-1)
                elif d=='S':
                    childCell=(currCell[0]+1,currCell[1])
                elif d=='N':
                    childCell=(currCell[0]-1,currCell[1])
                if childCell in explored:
                    continue
                frontier.append(childCell)
                explored.append(childCell)
                bfsPath[childCell] = currCell
                bSearch.append(childCell)
    # print(f'{bfsPath}')
    fwdPath={}
    cell=m._goal
    while cell!=(m.rows,m.cols):
        fwdPath[bfsPath[cell]]=cell
        cell=bfsPath[cell]
    return bSearch,bfsPath,fwdPath

def manhattan_distance(cell1,cell2):
    x1,y1=cell1
    x2,y2=cell2

    return abs(x1-x2) + abs(y1-y2)

def euclidean_distance(cell1, cell2):
    x1, y1 = cell1
    x2, y2 = cell2
    return ((x1-x2)**2+(y1-y2)**2)
    
def aStar():
    m=maze()
    # m.CreateMaze(loadMaze='maze--2023-03-10--13-30-20.csv')  # 5 X 5 Maze
    # m.CreateMaze(loadMaze='maze--2023-03-10--13-30-36.csv')  # 4 X 7 Maze
    m.CreateMaze(loadMaze='maze--2023-03-07--11-07-26.csv')  # 10 X 10 Maze

    start_time = timeit.default_timer()
    start=(m.rows,m.cols)
    g_score={cell:float('inf') for cell in m.grid}
    g_score[start]=0
    f_score={cell:float('inf') for cell in m.grid}
    f_score[start]=manhattan_distance(start,(1,1))

    open=PriorityQueue()
    open.put((manhattan_distance(start,(1,1)),manhattan_distance(start,(1,1)),start))
    aPath={}
    while not open.empty():
        currCell=open.get()[2]
        if currCell==(1,1):
            break
        for d in 'ESNW':
            if m.maze_map[currCell][d]==True:
                if d=='E':
                    childCell=(currCell[0],currCell[1]+1)
                if d=='W':
                    childCell=(currCell[0],currCell[1]-1)
                if d=='N':
                    childCell=(currCell[0]-1,currCell[1])
                if d=='S':
                    childCell=(currCell[0]+1,currCell[1])

                temp_g_score=g_score[currCell]+1
                temp_f_score=temp_g_score+manhattan_distance(childCell,(1,1))

                if temp_f_score < f_score[childCell]:
                    g_score[childCell]= temp_g_score
                    f_score[childCell]= temp_f_score
                    open.put((temp_f_score,manhattan_distance(childCell,(1,1)),childCell))
                    aPath[childCell]=currCell
    fwdPath={}
    cell=(1,1)
    while cell!=start:
        fwdPath[aPath[cell]]=cell
        cell=aPath[cell]
    end_time = timeit.default_timer()
    memory_used = memory_usage()
    time_taken = end_time - start_time

    print("Memory used by my_function: {} bytes".format(memory_used))
    print("Execution time for my_function: {} seconds".format(time_taken))
    a=agent(m,footprints=True,filled=True)
    m.tracePath({a:fwdPath},delay=100)
    l=textLabel(m,'Length of Shortest Path',len(fwdPath)+1)
    m.run()    

def aStar2():
    m=maze()
    # m.CreateMaze(loadMaze='maze--2023-03-10--13-30-20.csv')  # 5 X 5 Maze
    # m.CreateMaze(loadMaze='maze--2023-03-10--13-30-36.csv')  # 4 X 7 Maze
    m.CreateMaze(loadMaze='maze--2023-03-07--11-07-26.csv')  # 10 X 10 Maze

    start_time = timeit.default_timer()
    start=(m.rows,m.cols)
    g_score={cell:float('inf') for cell in m.grid}
    g_score[start]=0
    f_score={cell:float('inf') for cell in m.grid}
    f_score[start]=euclidean_distance(start,(1,1))

    open=PriorityQueue()
    open.put((euclidean_distance(start,(1,1)),euclidean_distance(start,(1,1)),start))
    aPath={}
    while not open.empty():
        currCell=open.get()[2]
        if currCell==(1,1):
            break
        for d in 'ESNW':
            if m.maze_map[currCell][d]==True:
                if d=='E':
                    childCell=(currCell[0],currCell[1]+1)
                if d=='W':
                    childCell=(currCell[0],currCell[1]-1)
                if d=='N':
                    childCell=(currCell[0]-1,currCell[1])
                if d=='S':
                    childCell=(currCell[0]+1,currCell[1])

                temp_g_score=g_score[currCell]+1
                temp_f_score=temp_g_score+euclidean_distance(childCell,(1,1))

                if temp_f_score < f_score[childCell]:
                    g_score[childCell]= temp_g_score
                    f_score[childCell]= temp_f_score
                    open.put((temp_f_score,euclidean_distance(childCell,(1,1)),childCell))
                    aPath[childCell]=currCell
    fwdPath={}
    cell=(1,1)
    while cell!=start:
        fwdPath[aPath[cell]]=cell
        cell=aPath[cell]
    end_time = timeit.default_timer()
    memory_used = memory_usage()
    time_taken = end_time - start_time

    print("Memory used by my_function: {} bytes".format(memory_used))
    print("Execution time for my_function: {} seconds".format(time_taken))
    a=agent(m,footprints=True,filled=True)
    m.tracePath({a:fwdPath},delay=100)
    l=textLabel(m,'Length of Shortest Path',len(fwdPath)+1)
    m.run()


def aStarTest(m,start=None):
    if start is None:
        start=(m.rows,m.cols)
    open = PriorityQueue()
    open.put((manhattan_distance(start, m._goal), manhattan_distance(start, m._goal), start))
    aPath = {}
    g_score = {row: float("inf") for row in m.grid}
    g_score[start] = 0
    f_score = {row: float("inf") for row in m.grid}
    f_score[start] = manhattan_distance(start, m._goal)
    searchPath=[start]
    while not open.empty():
        currCell = open.get()[2]
        searchPath.append(currCell)
        if currCell == m._goal:
            break        
        for d in 'ESNW':
            if m.maze_map[currCell][d]==True:
                if d=='E':
                    childCell=(currCell[0],currCell[1]+1)
                elif d=='W':
                    childCell=(currCell[0],currCell[1]-1)
                elif d=='N':
                    childCell=(currCell[0]-1,currCell[1])
                elif d=='S':
                    childCell=(currCell[0]+1,currCell[1])

                temp_g_score = g_score[currCell] + 1
                temp_f_score = temp_g_score + manhattan_distance(childCell, m._goal)

                if temp_f_score < f_score[childCell]:   
                    aPath[childCell] = currCell
                    g_score[childCell] = temp_g_score
                    f_score[childCell] = temp_g_score + manhattan_distance(childCell, m._goal)
                    open.put((f_score[childCell], manhattan_distance(childCell, m._goal), childCell))


    fwdPath={}
    cell=m._goal
    while cell!=start:
        fwdPath[aPath[cell]]=cell
        cell=aPath[cell]
    return searchPath,aPath,fwdPath

def mdp_value_iteration():
    m=maze()
    # m.CreateMaze(loadMaze='maze--2023-03-10--13-30-20.csv')  # 5 X 5 Maze
    # m.CreateMaze(loadMaze='maze--2023-03-10--13-30-36.csv')  # 4 X 7 Maze
    m.CreateMaze(loadMaze='maze--2023-03-07--11-07-26.csv')  # 10 X 10 Maze

    start_time = timeit.default_timer()
    THETA = 0.001
    GAMMA = 0.9         

    start=(m.rows,m.cols)
    aPath = {}
    #Define all states
    all_states=[]
    for i in range(1,m.rows+1):
        for j in range(1,m.cols+1):
                all_states.append((i,j))

    #Define rewards for all states
    rewards = {}
    for i in all_states:
       if i == (m.rows,m.cols):
            rewards[i] = 1
       else:
            rewards[i] = 0

    #Dictionnary of possible actions. We have two "end" states (1,2 and 2,2)
    actions = getActions(m)
    print(actions)
 
    #Define an initial policy
    policy={}
    for s in actions.keys():
        policy[s] = np.random.choice(actions[s])

    #Define initial value function 
    V={}
    for s in all_states:
        if s in actions.keys():
            V[s] = 0
        if s == (m.rows,m.cols):
            V[s]= 1
            
    '''==================================================
    Value Iteration
    =================================================='''

    iteration = 0

    while True:
        biggest_change = 0
        for s in all_states:  
            if(s == (m.rows,m.cols)):
                continue          
            if s in V:
                
                old_v = V[s]
                new_v = 0
                
                for a in actions[s]:
                    if a == 'U':
                        nxt = [s[0]-1, s[1]]
                    if a == 'D':
                        nxt = [s[0]+1, s[1]]
                    if a == 'L':
                        nxt = [s[0], s[1]-1]
                    if a == 'R':
                        nxt = [s[0], s[1]+1]

                    nxt = tuple(nxt)
                    # act = tuple(act)
                    v = rewards[s] + (GAMMA * (V[nxt] )) 
                    if v > new_v: #Is this the best action so far? If so, keep it
                        new_v = v
                        policy[s] = a

        #Save the best of all actions for the state                                
                V[s] = new_v
                biggest_change = max(biggest_change, np.abs(old_v - V[s]))

                
    #See if the loop should stop now         
        if biggest_change < THETA:
            for k in policy:
                print(str(k) + "->"+ str(policy[k])+ "   Value ->"+ str(V[k]))
            break
        iteration += 1
    currCell = (1,1)
    while(True):

        if(currCell == (m.rows,m.cols)):
            break
        if policy[currCell]=='R':
                childCell=(currCell[0],currCell[1]+1)
        if policy[currCell]=='L':
                childCell=(currCell[0],currCell[1]-1)
        if policy[currCell]=='U':
                childCell=(currCell[0]-1,currCell[1])
        if policy[currCell]=='D':
                childCell=(currCell[0]+1,currCell[1])

        aPath[childCell]=currCell
        currCell = childCell
    # fwdPath={}
    # cell=(1,1)
    # while cell!=start:
    #     fwdPath[aPath[cell]]=cell
    #     cell=aPath[cell]
    end_time = timeit.default_timer()
    memory_used = memory_usage()
    time_taken = end_time - start_time

    print("Memory used by my_function: {} bytes".format(memory_used))
    print("Execution time for my_function: {} seconds".format(time_taken))
    a=agent(m,footprints=True,filled=True)
    m.tracePath({a:aPath},delay=100)
    l=textLabel(m,'Length of Shortest Path',len(aPath)+1)
    m.run()

def mdp_policy_iteration():
    m=maze()
    # m.CreateMaze(loadMaze='maze--2023-03-10--13-30-20.csv')  # 5 X 5 Maze
    # m.CreateMaze(loadMaze='maze--2023-03-10--13-30-36.csv')  # 4 X 7 Maze
    m.CreateMaze(loadMaze='maze--2023-03-07--11-07-26.csv')  # 10 X 10 Maze

    start_time = timeit.default_timer()
    THETA = 0.001
    GAMMA = 0.9         

    start=(m.rows,m.cols)
    aPath = {}
    #Define all states
    all_states=[]
    for i in range(1,m.rows+1):
        for j in range(1,m.cols+1):
                all_states.append((i,j))

    #Define rewards for all states
    rewards = {}
    for i in all_states:
       if i == (m.rows,m.cols):
            rewards[i] = 1
       else:
            rewards[i] = 0

    #Dictionnary of possible actions. We have two "end" states (1,2 and 2,2)
    actions = getActions(m)

    #Define an initial policy
    policy={}
    for s in actions.keys():
        policy[s] = np.random.choice(actions[s])

    #Define initial value function 
    V={}
    for s in all_states:
        if s in actions.keys():
            V[s] = 0
        if s == (m.rows,m.cols):
            V[s]= 1
            
    '''==================================================
    Policy Iteration
    =================================================='''

    
    policychanged = True
    while(policychanged):
        iteration = 0
        policychanged = False
        while True:
            biggest_change = 0
            for s in all_states:  
                if(s == (m.rows,m.cols)):
                    continue          
                if s in V:
                    
                    old_v = V[s]
                    new_v = 0
                    
                    
                    if policy[s] == 'U':
                            nxt = [s[0]-1, s[1]]
                    if policy[s] == 'D':
                            nxt = [s[0]+1, s[1]]
                    if policy[s] == 'L':
                            nxt = [s[0], s[1]-1]
                    if policy[s] == 'R':
                            nxt = [s[0], s[1]+1]

                    nxt = tuple(nxt)
                        # act = tuple(act)
                    v = rewards[s] + (GAMMA * (V[nxt] )) 
                    if v > new_v: #Is this the best action so far? If so, keep it
                        new_v = v
                        # policy[s] = a

            #Save the best of all actions for the state                                
                    V[s] = new_v
                    biggest_change = max(biggest_change, np.abs(old_v - V[s]))

                    
        #See if the loop should stop now         
            if biggest_change < THETA:
                break
            iteration += 1
        for temp_s in all_states:
            if temp_s == (m.rows,m.cols):
                continue
            vtemp = {}
            for a in actions[temp_s]:
                    if a == 'U':
                        nxt = [temp_s[0]-1, temp_s[1]]
                        vtemp['U'] = V[tuple(nxt)]
                    if a == 'D':
                        nxt = [temp_s[0]+1, temp_s[1]]
                        vtemp['D'] = V[tuple(nxt)]
                    if a == 'L':
                        nxt = [temp_s[0], temp_s[1]-1]
                        vtemp['L'] = V[tuple(nxt)]
                    if a == 'R':
                        nxt = [temp_s[0], temp_s[1]+1]
                        vtemp['R'] = V[tuple(nxt)]

            best_action = max(vtemp, key=vtemp.get)   

            if(best_action != policy[temp_s]):
                policy[temp_s] = best_action 
                policychanged = True
                
        print("Iteration -> "+ str(iteration))
    currCell = (1,1)
    while(True):

        if(currCell == (m.rows,m.cols)):
            break
        if policy[currCell]=='R':
                childCell=(currCell[0],currCell[1]+1)
        if policy[currCell]=='L':
                childCell=(currCell[0],currCell[1]-1)
        if policy[currCell]=='U':
                childCell=(currCell[0]-1,currCell[1])
        if policy[currCell]=='D':
                childCell=(currCell[0]+1,currCell[1])

        aPath[childCell]=currCell
        currCell = childCell
    # fwdPath={}
    # cell=(1,1)
    # while cell!=start:
    #     fwdPath[aPath[cell]]=cell
    #     cell=aPath[cell]
    end_time = timeit.default_timer()
    memory_used = memory_usage()
    time_taken = end_time - start_time

    print("Memory used by my_function: {} bytes".format(memory_used))
    print("Execution time for my_function: {} seconds".format(time_taken))
    a=agent(m,footprints=True,filled=True)
    m.tracePath({a:aPath},delay=100)
    l=textLabel(m,'Length of Shortest Path',len(aPath)+1)
    m.run()

def memory_usage():
    memory_usage = 0
    for obj in gc.get_objects():
        memory_usage += sys.getsizeof(obj)
    return memory_usage

def getActions(m):
    actions = {}
    for i in m.grid:
        if(i == (m.rows,m.cols)):
            continue
        actions[i]=()
        for d in "ESNW":
            if(m.maze_map[i][d] == True):
                if(d=="E"):
                    actions[i] = (*actions[i],"R")
                if(d=="S"):
                    actions[i] = (*actions[i],"D")
                if(d=="W"):
                    actions[i] = (*actions[i],"L")
                if(d=="N"):
                    actions[i] = (*actions[i],"U")

    return actions                    

def CompareSearchAlgos(m):
    
    
    searchPath,dfsPath,fwdDFSPath=DFSTest(m)
    bSearch,bfsPath,fwdBFSPath=BFSTest(m)
    asearchPath,aPath,fwdPath=aStarTest(m)

    # l=textLabel(m,'A-Star Path Length',len(fwdPath)+1)
    textLabel(m,'DFS Path Length',len(fwdDFSPath)+1)
    textLabel(m,'BFS Path Length',len(fwdBFSPath)+1)
    l=textLabel(m,'A-Star Search Length',len(asearchPath)+1)
    textLabel(m,'DFS Search Length',len(searchPath)+1)
    textLabel(m,'BFS Search Length',len(bSearch)+1)

    a=agent(m,footprints=True,color=COLOR.cyan,filled=True)
    b=agent(m,footprints=True,color=COLOR.yellow)
    c = agent (m, footprints=True, color=COLOR.red)
    m.tracePath({a:fwdBFSPath},delay=100)
    m.tracePath({b:fwdDFSPath},delay=100)
    m.tracePath({c:fwdPath},delay=100)



    t1=timeit.timeit(stmt='DFS(m)',number=1000,globals=globals())
    t2=timeit.timeit(stmt='BFS(m)',number=1000,globals=globals())
    t3=timeit.timeit(stmt='aStar(m)',number=1000,globals=globals())

    textLabel(m,'DFS Time',t1)
    textLabel(m,'BFS Time',t2)
    textLabel(m,'A* Time',t3)

    m.run()



if __name__ == "__main__":

    mdp_policy_iteration()
   



