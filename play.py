from keras.models import load_model
import cv2
import numpy as np
from random import choice

REV_CLASS_MAP = {
    0: "rock",
    1: "paper",
    2: "scissors",
}

def mapper(val):
    return REV_CLASS_MAP[val]

def calculate_winner(move1, move2):
    if move1 == move2:
        return "Tie"

    if move1 == "rock":
        if move2 == "scissors":
            return "User"
        if move2 == "paper":
            return "PC"

    if move1 == "paper":
        if move2 == "rock":
            return "User"
        if move2 == "scissors":
            return "PC"

    if move1 == "scissors":
        if move2 == "paper":
            return "User"
        if move2 == "rock":
            return "PC"

start = False
Final = False
model = load_model("my_model.h5")
cap = cv2.VideoCapture(0)
prev_move = None
count_player = 0
count_pc = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue       
    # rectangle for user to play
    cv2.rectangle(frame, (0, 200), (300, 500), (255, 255, 255), 2)
    # rectangle for computer to play
    cv2.rectangle(frame, (350, 200), (650, 500), (255, 255, 255), 2)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "User: ", (0, 180), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "PC: ", (350, 180), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    
    if not start:
        cv2.putText(frame, "a start, q quit", (0, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "User: "+str(count_player), (0, 30), font, 1, (255, 215, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "PC: "+str(count_pc), (300, 30), font, 1, (255, 215, 0), 2, cv2.LINE_AA)
    
    
    # extract the region of image within the user rectangle
    roi = frame[200:500, 0:300]
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (64, 64))
    if start:
        # predict the move made
        pred = model.predict(np.array([img]))
        move_code = np.argmax(pred[0])
        user_move_name = mapper(move_code)

        # predict the winner (human vs computer)
        if prev_move != user_move_name:
            if user_move_name != "none":
                computer_move_name = choice(['rock', 'paper', 'scissors'])
                winner = calculate_winner(user_move_name, computer_move_name)
                if winner=='User' and count_player<10 and Final == False:
                    count_player = count_player+1
                if winner=='PC' and count_pc<10 and Final == False:
                    count_pc = count_pc+1
            else:
                computer_move_name = "none"
                winner = "Waiting..."
        prev_move = user_move_name
        
        if count_pc==10:
            cv2.putText(frame, "Final Winner: PC", (50, 100), font, 2, (0, 0, 255), 4, cv2.LINE_AA)
            cv2.putText(frame, "a start, q quit", (50, 135), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            Final = True
        if count_player==10:
            cv2.putText(frame, "Final Winner: User", (50, 100), font, 2, (0, 0, 255), 4, cv2.LINE_AA)
            cv2.putText(frame, "a start, q quit", (50, 135), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            Final = True
        
        # display the information
        if Final == False:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, "User: " + user_move_name, (0, 180), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "PC: " + computer_move_name, (350, 180), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "Winner: " + winner, (50, 100), font, 2, (0, 0, 255), 4, cv2.LINE_AA)
    
        if computer_move_name != "none" and Final == False:
            icon = cv2.imread("images/{}.png".format(computer_move_name))
            icon = cv2.resize(icon, (290, 280))
            frame[200:500, 350:650] = icon
        
    cv2.imshow("Rock Paper Scissors", frame)

    k = cv2.waitKey(10)
    if k == ord('a'):
        start = not start
        count_player = 0
        count_pc = 0
        Final = False 
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
