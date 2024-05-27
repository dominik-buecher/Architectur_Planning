import tkinter as tk
import random
import torch

class GridWindow:
    def __init__(self, root, rows, cols, num_cows):    
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
            cow = tk.Canvas(self.root, width=20, height=20, bg="white", highlightthickness=0)
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


    def move_mower_abs(self, row, col):
        # Prüfe, ob die nächste Position gültig ist
        if 0 <= row < self.rows and 0 <= col < self.cols:
            current_row, current_col = self.mower.grid_info()["row"], self.mower.grid_info()["column"]
            
            self.cows

            if self.cells[row][col]["bg"] == "green":
                self.cells[row][col]["bg"] = "#006400"  # Ändere die Farbe auf "dunkelgrün"

            # überprüfe ob auf dem Feld schon eine kuh steht, falls ja bleibt der rasenmäher auf dem gleichen feld stehen
            for cow in self.cows:
                cow_row = cow["cow"].grid_info()["row"]
                cow_col = cow["cow"].grid_info()["column"]
                if (row == cow_row) and (col == cow_col):
                    row = current_row
                    col = current_col

            # Setze die Position des Rasenmähers auf das neue Feld
            self.mower.grid(row=int(row), column=int(col))




    def get_state(self):

        tensor = torch.zeros(self.rows, self.cols, 2)
        mower_row = self.mower.grid_info()["row"]
        mower_col = self.mower.grid_info()["column"]
        
        # Setze 1 für Rasenmäherposition
        tensor[mower_row, mower_col, 0] = 1
        
        cows_row = []
        cows_col = []
        for cow in self.cows:
            cow_row = cow["cow"].grid_info()["row"]
            cow_col = cow["cow"].grid_info()["column"]
            cows_row.append(cow_row)
            cows_col.append(cow_col)
            # Setze 1 für Kuhpositionen
            #tensor[cow_row, cow_col, 0] = 1
        
        cows_pos = torch.tensor([cows_row, cows_col])

        target_row = self.target.grid_info()["row"]
        target_col = self.target.grid_info()["column"]
        # Setze 1 für Zielposition
        #tensor[target_row, target_col, 0] = 1
               
        # Füge den Besuchsstatus der Felder hinzu
        for row in range(self.rows):
            for col in range(self.cols):
                if self.cells[row][col]["bg"] == "#006400":
                    tensor[row, col, 1] = 1

        return tensor, cows_pos


    def get_future_state(self, action):
        # {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right', 4: 'Stay'}
        
        mower_row = self.mower.grid_info()["row"]
        mower_col = self.mower.grid_info()["column"]        
        temp_row = self.mower.grid_info()["row"]
        temp_col = self.mower.grid_info()["column"] 


        if action == 0:  # Move Up
            mower_row = max(0, mower_row - 1)
        elif action == 1:  # Move Down
            mower_row = min(self.rows - 1, mower_row + 1)
        elif action == 2:  # Move Left
            mower_col = max(0, mower_col - 1)
        elif action == 3:  # Move Right
            mower_col = min(self.cols - 1, mower_col + 1)
        # Bei der Aktion 4 ('Stay') bleibt der Rasenmäher auf dem aktuellen Feld stehen.
        
        if (mower_row == self.target.grid_info()["row"]) and (mower_col == self.target.grid_info()["column"]) and (temp_row == 13) and (temp_col == 14):
            mower_row = temp_row
            mower_col = temp_col
        # Falls sich eine kuh auf dem momentanen feld befindet, bleibt der mäher da stehen wo er ist
        for cow in self.cows:
            cow_row = cow["cow"].grid_info()["row"]
            cow_col = cow["cow"].grid_info()["column"]
            if (mower_row == cow_row) and (mower_col == cow_col):
                mower_row = temp_row
                mower_col = temp_col

        return mower_row, mower_col


    def get_reward(self, future_row, future_col, action):
        reward = 0

        mower_row = self.mower.grid_info()["row"]
        mower_col = self.mower.grid_info()["column"]
        
        # überprüfe ob die nächste position mit einer kuh kollidieren würde
        cows_row = []
        cows_col = []
        for cow in self.cows:
            cow_row = cow["cow"].grid_info()["row"]
            cow_col = cow["cow"].grid_info()["column"]
            if (future_row == cow_row) and (future_col == cow_col):
                reward += 0

        if not((mower_row == future_row) and (mower_col == future_col) and (action ==  4)):
            if self.cells[future_row][future_col]["bg"] == "#006400": 
                reward -= 1
            else:
                if (action == 2) or (action == 3):
                    reward += 2

                if (action == 3) and (not(mower_row % 2)):
                    reward += 15
                elif (action == 2) and (mower_row % 2):
                    reward += 15
                else:
                    reward += 5
                        
        visited_feld_counter = 0
        # Füge den Besuchsstatus der Felder hinzu
        for row in range(self.rows):
            for col in range(self.cols):
                if self.cells[row][col]["bg"] == "#006400":
                    visited_feld_counter += 1

        target_row = self.target.grid_info()["row"]
        target_col = self.target.grid_info()["column"]

        if (future_row == target_row) and (future_col == target_col) and (action == 3):
            print("Geschafft!")
            reward += 30
            reward += visited_feld_counter * 0.01
        return reward


    def reset_green_fields(self):
        for row in range(self.rows):
            for col in range(self.cols):
                if self.cells[row][col]["bg"] == "#006400":
                    self.cells[row][col]["bg"] = "green"
        self.cells[0][0]["bg"] = "#006400"


    def reset_to_initial_state(self, pos_cows):
        # Setze die Hintergrundfarbe aller Felder auf "green"
        for row in range(self.rows):
            for col in range(self.cols):
                self.cells[row][col]["bg"] = "green"
        self.cells[0][0]["bg"] = "#006400"
        
        i = 0
        for cow in self.cows:
            cow["cow"].grid(row=pos_cows[0, i].item(), column=pos_cows[1, i].item())
            i += 1

        self.mower.grid(row=0, column=0)
        #self.target.grid(row=self.rows, column=self.cols)



    def is_field_visited(self, state, rows, cols):
        # Extrahiere den Besuchsstatus-Teil des Zustandsvektors
        visited_status = state[-(rows * cols):]

        # Überprüfe, ob alle Felder (außer dem Ziel) besucht wurden
        return all(visited_status[:-2])  # Ignoriere die letzten beiden Werte (Ziel) beim Überprüfen

    
    def is_single_field_visited(self, state, row, col, rows, cols):
        # Extrahiere den Besuchsstatus-Teil des Zustandsvektors
        visited_status = state[-(rows * cols):]

        # Überprüfe, ob das spezifische Feld besucht wurde
        return visited_status[int(row) * cols + int(col)] == 1

    
    def is_90_percent_visited(self, state, rows, cols):
        # Extrahiere den Besuchsstatus-Teil des Zustandsvektors
        visited_status = state[-(rows * cols):]
        visited_count = sum(visited_status)
        total_fields = rows * cols
        return visited_count >= 0.9 * total_fields
    

        
def calculate_exponential_reward(cols, state, future_row, future_col, visited_status):
    # Überprüfe, ob der Rasenmäher auf ein Feld bewegt wurde, das bereits besucht wurde
    if visited_status[int(future_row) * cols + int(future_col)] == 1:
        return 0  # Kein zusätzlicher Reward, da das Feld bereits besucht wurde

    # Zähle die aufeinanderfolgenden nicht besuchten Felder
    consecutive_unvisited_count = 0
    for i in range(len(visited_status)):
        if visited_status[i] == 0:
            consecutive_unvisited_count += 1
        else:
            consecutive_unvisited_count = 0

    # Erhöhe den Reward exponentiell basierend auf der Anzahl der aufeinanderfolgenden nicht besuchten Felder
    reward_multiplier = 1.5
    exponential_reward = (reward_multiplier ** consecutive_unvisited_count) * 5

    return exponential_reward


def is_field_visited(state, rows, cols, mower_row, mower_col):
    # Extrahiere den Besuchsstatus-Teil des Zustandsvektors
    visited_status = state[-(rows * cols):]
    if visited_status[mower_row * cols + mower_col] == 1:
        return True

    return False


def main():
    root = tk.Tk()
    root.title("Grid Window with Cows, Mower, and Target")
    
    rows = 15
    cols = 15
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

