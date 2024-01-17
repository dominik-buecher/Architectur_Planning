import tkinter as tk
import random
import torch

class GridWindow:
    def __init__(self, root, rows, cols, num_cows, device):
        self.root = root
        self.rows = rows
        self.cols = cols
        self.cells = [[None for _ in range(cols)] for _ in range(rows)]
        self.device = device  # Store the device

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

    def create_cows(self):
        for _ in range(self.num_cows):
            cow_row, cow_col = self.get_random_empty_location()
            cow = tk.Canvas(self.root, width=20, height=20, bg="blue", highlightthickness=0)
            cow.grid(row=cow_row, column=cow_col)
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

        # Move state to GPU
        state = torch.tensor(state, dtype=torch.float32).to(self.device)

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

        row = int(state[0].item())
        col = int(state[1].item())

        row_cow.append(int(state[3].item()))
        col_cow.append(int(state[4].item()))

        row_cow.append(int(state[5].item()))
        col_cow.append(int(state[6].item()))

        row_cow.append(int(state[7].item()))
        col_cow.append(int(state[8].item()))

        row_cow.append(int(state[9].item()))
        col_cow.append(int(state[10].item()))

        if (
            (future_row == row_cow[0] and future_col == col_cow[0]) or
            (future_row == row_cow[1] and future_col == col_cow[1]) or
            (future_row == row_cow[2] and future_col == col_cow[2]) or
            (future_row == row_cow[3] and future_col == col_cow[3])
        ):
            reward = -10
        else:
            visited_status = state[-(self.rows * self.cols):].to(torch.bool)  # Convert to boolean tensor
            if visited_status[future_row.long() * self.cols + future_col.long()].item():
                reward = -5
            else:
                reward = 5

            if row == int(self.target.grid_info()["row"]) and col == int(self.target.grid_info()["column"]):
                alle_besucht = all(state[12:])
                if alle_besucht:
                    reward = 500

        return reward


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Detect GPU availability
    grid_window = GridWindow(root, rows, cols, num_cows, device)  # Pass the device to GridWindow

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
