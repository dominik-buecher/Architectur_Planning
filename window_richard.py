import tkinter as tk
import random


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
<<<<<<< HEAD
        self.cows = []
        for _ in range(self.num_cows):
            cow_row, cow_col = self.get_random_empty_location()
=======
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
>>>>>>> 668ad806788d672200d6c2e0cf7df7365afe12f8
            cow = tk.Canvas(self.root, width=20, height=20, bg="blue", highlightthickness=0)
            cow.grid(row=start_row, column=start_col)
            self.cows.append({"cow": cow, "direction": random.choice(["up", "down", "left", "right"])})
    
    def get_cow_positions(self):
        positions = []
        for c in self.cows:
            current_row = c["cow"].grid_info()["row"]
            current_col = c["cow"].grid_info()["column"]
            positions.append((current_row, current_col))
        assert(len(self.cows) == 2)
        return positions


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

            new_row = current_row
            new_col = current_col

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
                cow["direction"] = random.choice(["up", "down", "left", "right"])

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
        # Kopiere den aktuellen Zustand
        current_state = self.get_state()

        # Überprüfe, ob das Zielfeld bereits von einer Kuh besetzt ist
        if any(c["cow"].grid_info()["row"] == row and c["cow"].grid_info()["column"] == col for c in self.cows):
            # Wenn eine Kuh auf dem Zielfeld ist, kehre zum alten Zustand zurück und beende die Funktion
            self.mower.grid(row=current_state[0], column=current_state[1])
            return

        if self.cells[row][col]["bg"] == "green":
            self.cells[row][col]["bg"] = "#006400"  # Ändere die Farbe auf "dunkelgrün"

        # Setze den Rasenmäher auf das Zielzellenfeld
        self.mower.grid(row=row, column=col)


    def get_state(self):
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
        return state

    def get_future_state(self, current_state, action):
        # {0: 'Up', 1: 'Down', 2:'Left', 3: 'Right'}
        
        row = current_state[0]
        col = current_state[1]

        if action == 0:  # Move Up
            row = row - 1
            row = max(0, row)
        elif action == 1:  # Move Down
            row = row + 1
            row = min(self.rows - 1, row)
        elif action == 2:  # Move Left
            col = col - 1
            col = max(0, col)
        elif action == 3:  # Move Right
            col = col + 1
            col = min(self.cols - 1, col)

        # return row, col 
        return row, col 



    def get_reward(self, state, future_row, future_col):
        
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
            reward = -10

        visited_status = state[-(self.rows * self.cols):]

        # Überprüfe, ob das Feld an der Position des Rasenmähers besucht wurde
        if visited_status[future_row * self.cols + future_col] == 1:
            reward = -5
        else:
            reward = 5

        if row == self.target.grid_info()["row"] and col == self.target.grid_info()["column"]:

            alle_besucht = all(state[12:])

            # if alle_besucht is True:
            #     reward = 1000
            reward = 5000

        return reward
    

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
        for cow, initial_cow_position in zip(self.cows, initial_state[2:11:2]):
            cow["cow"].grid(row=initial_cow_position, column=initial_state[initial_cow_position + 1])

        # Setze die Position des Rasenmähers zurück
        self.mower.grid(row=initial_state[0], column=initial_state[1])



    



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
