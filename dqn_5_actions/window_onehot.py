import tkinter as tk
import random
import numpy as np
import torch
import torchvision.transforms.functional as F
import torch.nn.functional as FF

class GridWindow:
    def __init__(self, root, rows, cols, num_cows):
        
        self.state_size = 2 + 2 * num_cows + 2 * rows * cols  # 2 für Mower, 2 für jede Kuh, 2 für das Ziel, 2 für jedes Rasterfeld (One-Hot-kodiert)
        self.root = root
        self.rows = rows
        self.cols = cols
        self.cells = [[None for _ in range(cols)] for _ in range(rows)]

        self.cows = []
        self.num_cows = num_cows

        self.mower = None
        self.target = None

        self.create_grid()
        self.create_cows()
        self.create_mower()
        self.create_target()

        # print("target: ", self.target.grid_info()["row"])
        # print("target: ", self.target.grid_info()["column"])

    def create_grid(self):
        for row in range(self.rows):
            for col in range(self.cols):
                cell = tk.Canvas(self.root, width=20, height=20, bg="green", highlightthickness=0)
                cell.grid(row=row, column=col)
                self.cells[row][col] = cell

    def reset(self):
        for r in range(self.rows):
            for c in range(self.cols):
                self.cells[r][c]["bg"] = "green"
        self.create_mower()
        self.create_target()

    def create_cows(self):
        # Feste Startpositionen für die Kühe
        start_positions = [
            (self.rows // 2, self.cols // 2),
            (self.rows // 3, self.cols // 3),
            (2 * self.rows // 3, 2 * self.cols // 3),
            (self.rows // 4, 3 * self.cols // 4),
            (3 * self.rows // 4, self.cols // 4)
        ]

        for i in range(min(self.num_cows, len(start_positions))):
            start_row, start_col = start_positions[i]
            cow = tk.Canvas(self.root, width=20, height=20, bg="blue", highlightthickness=0)
            cow.grid(row=start_row, column=start_col)
            self.cows.append({"cow": cow, "direction": random.choice(["up", "down", "left", "right"])})

    def create_mower(self):
        self.mower = tk.Canvas(self.root, width=20, height=20, bg="red", highlightthickness=0)
        self.mower.grid(row=0, column=0)  
        self.mower_direction = "right"
        self.cells[0][0]["bg"] = "#006400" 

    def create_target(self):
        target_row = self.rows - 1
        target_col = self.cols - 1
        self.target = tk.Canvas(self.root, width=20, height=20, bg="gray", highlightthickness=0)
        self.target.grid(row=target_row, column=target_col)

    def get_random_empty_location(self):
        while True:
            row = random.randint(0, self.rows - 1)
            col = random.randint(0, self.cols - 1)
            if not any(cow["cow"].grid_info()["row"] == row and cow["cow"].grid_info()["column"] == col for cow in self.cows) \
                    and not (self.mower is not None and self.mower.grid_info()["row"] == row and self.mower.grid_info()["column"] == col) \
                    and not (self.target is not None and self.target.grid_info()["row"] == row and self.target.grid_info()["column"] == col):
                return row, col

    def move_cows(self):
        for cow in self.cows:
            direction = cow["direction"]
            current_row = cow["cow"].grid_info()["row"]
            current_col = cow["cow"].grid_info()["column"]

            direction = random.choice(["up", "down", "left", "right"])
            new_col = current_col
            new_row = current_row

            if direction == "up":
                new_row = max(0, current_row - 1)
            elif direction == "down":
                new_row = min(self.rows - 1, current_row + 1)
            elif direction == "left":
                new_col = max(0, current_col - 1)
            elif direction == "right":
                new_col = min(self.cols - 1, current_col + 1)

            if not any(c["cow"].grid_info()["row"] == new_row and c["cow"].grid_info()["column"] == new_col for c in self.cows) \
                    and not (self.mower is not None and self.mower.grid_info()["row"] == new_row and self.mower.grid_info()["column"] == new_col) \
                    and not (self.target is not None and self.target.grid_info()["row"] == new_row and self.target.grid_info()["column"] == new_col):
                cow["cow"].grid(row=new_row, column=new_col)
            else:
                cow["cow"].grid(row=current_row, column=current_col)








    def move_mower(self, event):
        current_row = self.mower.grid_info()["row"]
        current_col = self.mower.grid_info()["column"]

        new_row = current_row
        new_col = current_col

        if event.keysym == "Up":
            new_row = max(0, current_row - 1)
            self.mower_direction = "up"
        elif event.keysym == "Down":
            new_row = min(self.rows - 1, current_row + 1)
            self.mower_direction = "down"
        elif event.keysym == "Left":
            new_col = max(0, current_col - 1)
            self.mower_direction = "left"
        elif event.keysym == "Right":
            new_col = min(self.cols - 1, current_col + 1)
            self.mower_direction = "right"

        if not any(c["cow"].grid_info()["row"] == new_row and c["cow"].grid_info()["column"] == new_col for c in self.cows):
            self.mower.grid(row=new_row, column=new_col)
            if self.cells[new_row][new_col]["bg"] == "green":
                self.cells[new_row][new_col]["bg"] = "#006400"

    
    # def move_mower_abs(self, row, col, coordinates):
    #     # Kopiere den aktuellen Zustand
    #     current_state = self.get_state(coordinates)

    #     # Überprüfe, ob das Zielfeld bereits von einer Kuh besetzt ist
    #     if any(c["cow"].grid_info()["row"] == row and c["cow"].grid_info()["column"] == col for c in self.cows):
    #         # Wenn eine Kuh auf dem Zielfeld ist, kehre zum alten Zustand zurück und beende die Funktion
    #         self.mower.grid(row=current_state[0], column=current_state[1])
    #         return

    #     if self.cells[row][col]["bg"] == "green":
    #         self.cells[row][col]["bg"] = "#006400"  # Ändere die Farbe auf "dunkelgrün"

    #     # Setze den Rasenmäher auf das Zielzellenfeld
    #     self.mower.grid(row=row, column=col)

    def move_mower_abs(self, row, col, coordinates):
        # Prüfe, ob die nächste Position gültig ist
        if 0 <= row < self.rows and 0 <= col < self.cols:
            # Überprüfe, ob das Feld bereits besucht wurde
            if self.cells[row.item()][col.item()]["bg"] == "green":
                # Setze die Hintergrundfarbe des aktuellen Feldes auf "green"
                current_row, current_col = coordinates[0].item(), coordinates[1].item()
                self.cells[current_row][current_col]["bg"] = "green"

                # Setze die Hintergrundfarbe des Zielfeldes auf "#006400"
                self.cells[self.target.grid_info()["row"]][self.target.grid_info()["column"]]["bg"] = "#006400"

                # Setze die Hintergrundfarbe des neuen Feldes auf "#FFFF00"
                self.cells[row.item()][col.item()]["bg"] = "#FFFF00"

                # Setze die Position des Rasenmähers auf das neue Feld
                self.mower.grid(row=row.item(), column=col.item())




    def get_state(self, coordinates):
        state = []

        # Füge Positionen des Rasenmähers, der Kühe und des Ziels hinzu
        state.extend([self.mower.grid_info()["row"], self.mower.grid_info()["column"]])
        for cow in self.cows:
            state.extend([cow["cow"].grid_info()["row"], cow["cow"].grid_info()["column"]])
        state.extend([self.target.grid_info()["row"], self.target.grid_info()["column"]])
        
        # Füge den Besuchsstatus der Felder hinzu
        for row in range(self.rows):
            for col in range(self.cols):
                if self.cells[row][col]["bg"] == "#006400":
                    state.append(1)  # Besucht
                else:
                    state.append(0)  # Nicht besucht

        # Konvertiere die Liste in einen Tensor und wende die One-Hot-Kodierung an
        state_tensor = torch.tensor(state[:coordinates])
        print("state_tensor: ", state_tensor)
        field_states_tensor = torch.tensor(state[coordinates:])
        state_one_hot = FF.one_hot(state_tensor.long(), num_classes=self.rows).float()

        state_one_hot = state_one_hot.view((1+1+self.num_cows)*2*self.cols)
        state_tensor = torch.cat((state_one_hot, field_states_tensor))

        return state_one_hot

    def get_future_state(self, current_state, action):
        # {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right', 4: 'Stay'}
        row = current_state[0]
        col = current_state[1]

        if action == 0:  # Move Up
            row = max(0, row - 1)
        elif action == 1:  # Move Down
            row = min(self.rows - 1, row + 1)
        elif action == 2:  # Move Left
            col = max(0, col - 1)
        elif action == 3:  # Move Right
            col = min(self.cols - 1, col + 1)
        # Bei der Aktion 4 ('Stay') bleibt der Rasenmäher auf dem aktuellen Feld stehen.

        return row, col


    def get_reward(self, state, future_row, future_col, action, action_counter, num_cows, consecutive_unvisited_count):
        reward = 0
        row_cow = []
        col_cow = []

        row = state[0]
        col = state[1]

        row_cow = []
        col_cow = []

        row = state[0]
        col = state[1]



        for i in range(2, len(state) - 2, 2):
            row_cow.append(state[i])
            col_cow.append(state[i + 1])
        # [position of mower row, position of mower col, position of cow1 row, position of cow1 col, position of cow2 row, position of cow2 col, position of cow3 row, position of cow3 col, position of cow4 row, position of cow4 col, position of cow5 row, position of cow5 col, 1 if we have visitied this field and 0 if not ]
        # state = [mower_row, mower_col, cow1_row, cow1_col, cow2_row, cow2_col, ..., target_row, target_col, 0, 1, 0, 1, ...]

        if (future_row, future_col) in zip(row_cow, col_cow):
            reward += -5

        visited_status = state[-(self.rows * self.cols):]

        if not (state[0] == future_row and state[1] == future_col):
            # Überprüfe, ob das Feld an der Position des Rasenmähers besucht wurde
            if visited_status[future_row * self.cols + future_col] == 1:
                reward += -5
                consecutive_unvisited_count = 0
            else:
                if action == 2 or action == 3:
                    reward += 10
                else:
                    reward += 5
                #reward += 50
                reward += calculate_exponential_reward(self.cols, state, future_row, future_col, visited_status)
                # Erhöhe den Status der aufeinanderfolgenden unbesuchten Felder
                consecutive_unvisited_count += 1
                if row % 2 != 0:
                    if row == future_row and col + 1 == future_col:
                        reward += 20
                else:
                    if row == future_row and col - 1 == future_col:
                        reward += 20


                        
        positionen = 2 + (2 * num_cows)
        alle_besucht = all(state[positionen:])

        if alle_besucht is True:
            reward += 100
        if self.is_90_percent_visited(state, self.rows, self.cols):
            if row == self.target.grid_info()["row"] and col == self.target.grid_info()["column"]:

                base_reward = 10
                max_multiplier = 30
                multiplier_threshold = 1000

                # Berechnen Sie den Multiplikator basierend auf dem action_counter
                multiplier = max(1, max_multiplier - (action_counter / multiplier_threshold))

                # Berechnen Sie den endgültigen Reward
                reward += base_reward * multiplier
                print("reward: ", reward)


        # # Belohne, wenn der Roboter eine Zeile nach der anderen abfährt
        # if action == 3 and col == self.cols - 1:
        #     # Überprüfe, ob alle Felder in der aktuellen Zeile abgefahren wurden
        #     if all(visited_status[row * self.cols : (row + 1) * self.cols]):
        #         reward += 500  # Belohne das vollständige Abfahren der aktuellen Zeile
        # elif action == 2 and col == 0:
        #     # Überprüfe, ob alle Felder in der aktuellen Zeile abgefahren wurden
        #     if all(visited_status[row * self.cols : (row + 1) * self.cols]):
        #         reward += 500  # Belohne das vollständige Abfahren der aktuellen Zeile


        return reward, consecutive_unvisited_count
    

    def reset_green_fields(self):
        for row in range(self.rows):
            for col in range(self.cols):
                if self.cells[row][col]["bg"] == "#006400":
                    self.cells[row][col]["bg"] = "green"
        self.cells[0][0]["bg"] = "#006400"


    def reset_to_initial_state(self, initial_state):
        # Setze die Hintergrundfarbe aller Felder auf "green"
        for row in range(self.rows):
            for col in range(self.cols):
                self.cells[row][col]["bg"] = "green"
        self.cells[0][0]["bg"] = "#006400"

        # Setze die Positionen der Kühe zurück
        cow_positions = initial_state[:self.num_cows*2]
        for cow, cow_position in zip(self.cows, cow_positions):
            cow["cow"].grid(row=int(cow_position.item()), column=int(cow_positions[cow_position + 1].item()))

        # Setze die Position des Rasenmähers zurück
        mower_position = initial_state[self.num_cows*2:self.num_cows*2 + 2]
        self.mower.grid(row=int(mower_position[0].item()), column=int(mower_position[1].item()))



    def is_field_visited(self, state, rows, cols):
        # Extrahiere den Besuchsstatus-Teil des Zustandsvektors
        visited_status = state[-(rows * cols):]

        # Überprüfe, ob alle Felder (außer dem Ziel) besucht wurden
        return all(visited_status[:-2])  # Ignoriere die letzten beiden Werte (Ziel) beim Überprüfen

    def is_single_field_visited(self, state, row, col, rows, cols):
        # Extrahiere den Besuchsstatus-Teil des Zustandsvektors
        visited_status = state[-(rows * cols):]

        # Überprüfe, ob das spezifische Feld besucht wurde
        return visited_status[row * cols + col] == 1
    
    def is_90_percent_visited(self, state, rows, cols):
        # Extrahiere den Besuchsstatus-Teil des Zustandsvektors
        visited_status = state[-(rows * cols):]

        # Zähle die Anzahl der besuchten Felder
        visited_count = sum(visited_status)

        # Berechne die Gesamtanzahl der Felder im Raster
        total_fields = rows * cols

        # Überprüfe, ob 90% oder mehr der Felder besucht wurden
        return visited_count >= 0.9 * total_fields
        
def calculate_exponential_reward(cols, state, future_row, future_col, visited_status):
        # Überprüfe, ob der Rasenmäher auf ein Feld bewegt wurde, das bereits besucht wurde
        if visited_status[future_row * cols + future_col] == 1:
            return 0  # Kein zusätzlicher Reward, da das Feld bereits besucht wurde

        # Zähle die aufeinanderfolgenden nicht besuchten Felder
        consecutive_unvisited_count = 0
        for i in range(len(visited_status)):
            if visited_status[i] == 0:
                consecutive_unvisited_count += 1
            else:
                consecutive_unvisited_count = 0

        # Erhöhe den Reward exponentiell basierend auf der Anzahl der aufeinanderfolgenden nicht besuchten Felder
        reward_multiplier = 1.5  # Experimentiere mit verschiedenen Werten für den Multiplikator
        exponential_reward = (reward_multiplier ** consecutive_unvisited_count) * 5  # Grundreward von 10, kann angepasst werden

        return exponential_reward


def is_field_visited(state, rows, cols, mower_row, mower_col):
    # Extrahiere den Besuchsstatus-Teil des Zustandsvektors
    visited_status = state[-(rows * cols):]

    # Überprüfe, ob das Feld an der Position des Rasenmähers oder des Ziels bereits besucht wurde
    if visited_status[mower_row * cols + mower_col] == 1:
        return True

    return False

def main():
    root = tk.Tk()
    root.title("Grid Window with Cows, Mower, and Target")
    
    rows = 20
    cols = 20
    num_cows = 5

    grid_window = GridWindow(root, rows, cols, num_cows)

    root.bind("<Up>", grid_window.move_mower)
    root.bind("<Down>", grid_window.move_mower)
    root.bind("<Left>", grid_window.move_mower)
    root.bind("<Right>", grid_window.move_mower)

    def update():
        grid_window.move_cows()
        root.after(500, update) 

    update()

    root.mainloop()

if __name__ == "__main__":
    main()
