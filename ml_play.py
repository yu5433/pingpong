from mlgame.communication import ml as comm
import os.path as path
import pickle
import numpy as np
def ml_loop(side: str):
    """
    The main loop for the machine learning process
    The `side` parameter can be used for switch the code for either of both sides,
    so you can write the code for both sides in the same script. Such as:
    ```python
    if side == "1P":
        ml_loop_for_1P()
    else:
        ml_loop_for_2P()
    ```
    @param side The side which this script is executed for. Either "1P" or "2P".
   
    if side == "1P":
        ml_loop_for_1P()
    else:
        ml_loop_for_2P() """
    # === Here is the execution order of the loop === #
    # 1. Put the initialization code here
    ball_served = False
    """
    if side == "1P":
        filename = path.join(path.dirname(__file__),'save', 'player1.pickle')
        #filename = "MLGame-master\games\pingpong\save\player1.pickle"
        with open(filename, 'rb') as file:
            clf = pickle.load(file)
    else:
        
        filename = path.join(path.dirname(__file__),'save', 'player2.pickle')
        #filename = "MLGame-master\games\pingpong\save\player2.pickle"
        with open(filename, 'rb') as file:
            clf = pickle.load(file)
    """

    filename = "player1_5.pickle"
    with open(filename, 'rb') as file:
        clf = pickle.load(file)
    # 2. Inform the game process that ml process is ready
    comm.ml_ready()
    s = [85,250]
    def get_direction(VectorX, VectorY):
        
        if(VectorX>=0 and VectorY>=0):
            return 0
        elif(VectorX>0 and VectorY<0):
            return 1
        elif(VectorX<0 and VectorY>0):
            return 2
        else:
            return 3
    
    def get_Vx(ball_x,ball_y,ball_pre_x,ball_pre_y):
        VectorX = ball_x - ball_pre_x
        return VectorX

    def get_Vy(ball_x,ball_y,ball_pre_x,ball_pre_y):
        VectorY = ball_y - ball_pre_y
        return VectorY

    # 3. Start an endless loop
    while True:
        # 3.1. Receive the scene information sent from the game process
        scene_info = comm.recv_from_game()
        feature = []
        vector = []
        vector.append(scene_info['ball_speed'][0])
        vector.append(scene_info['ball_speed'][1])

        feature.append(scene_info['ball'][0])
        feature.append(scene_info['ball'][1])
        feature.append(scene_info['platform_1P'][0])
        #feature.append(vector[0])
        #feature.append(vector[1])
        feature.append(get_direction(vector[0], vector[1]))

        s = [vector[0], vector[1]]
        feature = np.array(feature)
        feature = feature.reshape((-1,4))
        #print(feature)
        #print(feature)
        # 3.2. If either of two sides wins the game, do the updating or
        #      resetting stuff and inform the game process when the ml process
        #      is ready.
        if scene_info["status"] != "GAME_ALIVE":
            # Do some updating or resetting stuff
            ball_served = False

            # 3.2.1 Inform the game process that
            #       the ml process is ready for the next round
            comm.ml_ready()
            continue

        # 3.3 Put the code here to handle the scene information

        # 3.4 Send the instruction for this frame to the game process
        if not ball_served:
            comm.send_to_game({"frame": scene_info["frame"], "command": "SERVE_TO_LEFT"})
            ball_served = True
        else:
            y = clf.predict(feature)
            #print(y)
            if y == 0:
                comm.send_to_game({"frame": scene_info["frame"], "command": "NONE"})
            #    comm.send_instruction(scene_info.frame, PlatformAction.NONE)
               # print('NONE')
            elif y == 1:
            #    comm.send_instruction(scene_info.frame, PlatformAction.MOVE_LEFT)
                comm.send_to_game({"frame": scene_info["frame"], "command": "MOVE_LEFT"})
                #print('LEFT')
            elif y == 2:
            #    comm.send_instruction(scene_info.frame, PlatformAction.MOVE_RIGHT)
                comm.send_to_game({"frame": scene_info["frame"], "command": "MOVE_RIGHT"})
                #print('RIGHT')

            
