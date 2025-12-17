import pyray as pr
from random import randint


# Settings ------------
CELL_SIZE = 35 
GRID_UNITS = 25
WINDOW_HEIGHT = CELL_SIZE * GRID_UNITS
WINDOW_WIDTH = CELL_SIZE * GRID_UNITS
MOVE_INTERVAL = 0.2 # Snake moves every 0.2 seconds

# States -------------
GAME_ACTIVE = 0
GAME_OVER = 1
game_state = GAME_ACTIVE

# Main Variables --
snake_body = []
snake_direction = pr.Vector2(0, 0) # Initial direction (will be set to right after init)
food_pos = pr.Vector2(0, 0)
time_since_last_move = 0.0
score = 0
background = pr.Color(144, 238, 144, 200)
foreground = pr.Color(1, 50, 32, 225)

def initialize_game():
    """Sets up initial snake position, direction, and first food."""
    global snake_body, snake_direction, food_pos, game_state, score, time_since_last_move

    # Reset state
    game_state = GAME_ACTIVE
    score = 0
    time_since_last_move = 0.0

    # Initial Snake Body (Head at (5, 10), moving right)
    snake_body = [
        pr.Vector2(5, 10),
        pr.Vector2(4, 10),
        pr.Vector2(3, 10)
    ]
    # Initial Direction: Right
    snake_direction = pr.Vector2(1, 0) 
    
    # Place the first piece of food
    food_pos = place_food()

def place_food():
    """Generates a random, safe position for the food (not under the snake)."""
    while True:
        # Generate random coordinates within the grid limits (0 to GRID_UNITS - 1)
        rand_x = randint(0, GRID_UNITS - 1)
        rand_y = randint(0, GRID_UNITS - 1)
        new_pos = pr.Vector2(rand_x, rand_y)
        
        # Check if the new position overlaps with the snake's body
        is_safe = True
        for segment in snake_body:
            if new_pos.x == segment.x and new_pos.y == segment.y:
                is_safe = False
                break
        
        if is_safe:
            return new_pos # Found a safe spot!

def update_game():
    """Handles input, movement, growth, and collision checks."""
    global snake_direction, time_since_last_move, food_pos, game_state, score

    # --- 1. HANDLE INPUT (Check for opposite direction to prevent instant self-collision) ---
    new_direction = pr.Vector2(snake_direction.x, snake_direction.y)

    if pr.is_key_pressed(pr.KEY_D) or pr.is_key_pressed(pr.KEY_RIGHT):
        new_direction = pr.Vector2(1, 0)
    elif pr.is_key_pressed(pr.KEY_A) or pr.is_key_pressed(pr.KEY_LEFT):
        new_direction = pr.Vector2(-1, 0)
    elif pr.is_key_pressed(pr.KEY_W) or pr.is_key_pressed(pr.KEY_UP):
        # Y-coordinates decrease as you move up the screen
        new_direction = pr.Vector2(0, -1) 
    elif pr.is_key_pressed(pr.KEY_S) or pr.is_key_pressed(pr.KEY_DOWN):
        # Y-coordinates increase as you move down the screen
        new_direction = pr.Vector2(0, 1)

    # Check if the new direction is the opposite of the current direction
    # e.g., current (1, 0) and new (-1, 0) means (1 + -1) == 0 AND (0 + 0) == 0.1, 50, 32)
    if (new_direction.x + snake_direction.x) != 0 or \
       (new_direction.y + snake_direction.y) != 0:
        # Only update if the new direction is not the opposite
        snake_direction = new_direction

    # --- 2. MOVEMENT TIMING ---
    time_since_last_move += pr.get_frame_time()
    
    if time_since_last_move >= MOVE_INTERVAL:
        
        # Check if the snake is actually moving (has a direction)
        if snake_direction.x == 0 and snake_direction.y == 0:
            time_since_last_move = 0.0
            return

        # 3. CALCULATE NEW HEAD
        current_head = snake_body[0]
        new_head = pr.Vector2(current_head.x + snake_direction.x, current_head.y + snake_direction.y)
        
        # Add the new head to the front of the body list
        snake_body.insert(0, new_head) 

        # A) Wall Collision (Out of bounds)
        if new_head.x < 0:
            new_head.x = GRID_UNITS
        elif new_head.x > GRID_UNITS:
            new_head.x = 0
        elif new_head.y < 0:
            new_head.y = GRID_UNITS
        elif new_head.y > GRID_UNITS:
            new_head.y = 0

        # 4. CHECK FOR FOOD 
        if new_head.x == food_pos.x and new_head.y == food_pos.y:
            # Snake EATS: Do not remove the tail (snake grows)
            score += 10
            food_pos = place_food() # Place new food immediately
        else:
            # Snake MOVES: Remove the last segment (normal movement)
            snake_body.pop() 
            
        # 5. COLLISION CHECKS 
        
        #if new_head.x < 0 or new_head.x >= GRID_UNITS or \
        #       new_head.y < 0 or new_head.y >= GRID_UNITS:
        #game_state = GAME_OVER
            
        # B) Self-Collision (Head hits any part of the body *excluding* the new head itself)
        # We check from index 1 onwards
        for segment in snake_body[1:]:
            if new_head.x == segment.x and new_head.y == segment.y:
                game_state = GAME_OVER
                break # Collision detected, no need to check further

        # Reset the timer
        time_since_last_move = 0.0

def draw_game():
    """Renders the snake, food, and score to the screen."""
    pr.begin_drawing()
    pr.clear_background(background)
    
    # --- DRAW GRID (Optional, for visual help) ---
    # Draw vertical lines
    #for i in range(1, GRID_UNITS):
    #    pr.draw_line(i * CELL_SIZE, 0, i * CELL_SIZE, WINDOW_HEIGHT, pr.DARKGRAY)
    # Draw horizontal lines
    #for i in range(1, GRID_UNITS):
    #    pr.draw_line(0, i * CELL_SIZE, WINDOW_WIDTH, i * CELL_SIZE, pr.DARKGRAY)

    # --- DRAW FOOD  ---
    f_x = int(food_pos.x * CELL_SIZE)
    f_y = int(food_pos.y * CELL_SIZE)
    # Draw the food slightly smaller than the cell size to make it distinct
    pr.draw_rectangle(f_x + 3, f_y + 3, CELL_SIZE - 6, CELL_SIZE - 6, foreground)

    # --- DRAW SNAKE  ---
    # Iterate through the list of Vector2s (grid coordinates)
    for i, segment in enumerate(snake_body):
        # Convert grid coordinates (segment.x, segment.y) to pixel coordinates
        x_pos = int(segment.x * CELL_SIZE)
        y_pos = int(segment.y * CELL_SIZE)
        
        # Use a slightly different color for the head (i==0)
        #color = pr.LIME if i == 0 else foreground
        
        # Draw the segment
        pr.draw_rectangle(x_pos, y_pos, CELL_SIZE, CELL_SIZE, foreground)
        
        # Draw a small inner border to make segments look distinct
        pr.draw_rectangle_lines(x_pos, y_pos, CELL_SIZE, CELL_SIZE, background)

    # --- DRAW SCORE ---
    current_score = F"SCORE: {score}"
    score_w = pr.measure_text(current_score, 20)
    pr.draw_rectangle(0, 0, score_w + 20, 35, pr.Color(0, 0, 0, 150)) 
    pr.draw_text(f"SCORE: {score}", 10, 10, 20, background)

    # --- DRAW GAME OVER SCREEN ---
    if game_state == GAME_OVER:
        # Draw a translucent black overlay
        pr.draw_rectangle(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, pr.Color(0, 0, 0, 150)) 
        
        game_over_text = "GAME OVER!"
        score_text = f"Final Score: {score}"
        restart_text = "Press [R] to Restart or [ESC] to Quit"
        tips_text = "Use arrow keys or WASD to move"
        
        # Center the Game Over text
        text_w = pr.measure_text(game_over_text, 50)
        pr.draw_text(game_over_text, WINDOW_WIDTH // 2 - text_w // 2, WINDOW_HEIGHT // 2 - 60, 50, background)
        # Center the Score text
        score_w = pr.measure_text(score_text, 30)
        pr.draw_text(score_text, WINDOW_WIDTH // 2 - score_w // 2, WINDOW_HEIGHT // 2, 30, background)
        
        # Center the Restart instruction
        restart_w = pr.measure_text(restart_text, 20)
        pr.draw_text(restart_text, WINDOW_WIDTH // 2 - restart_w // 2, WINDOW_HEIGHT // 2 + 50, 20, foreground)

        tips_w = pr.measure_text(tips_text, 20)
        pr.draw_text(tips_text, WINDOW_WIDTH // 2 - tips_w // 2, WINDOW_HEIGHT // 2 + 75, 20, foreground)


    pr.end_drawing()

# --- MAIN EXECUTION ---
def main():
    """The main game loop setup and execution."""
    pr.init_window(WINDOW_WIDTH, WINDOW_HEIGHT, "Test")
    pr.set_target_fps(60)
    
    # Start the game
    initialize_game()

    # The Core Game Loop
    while not pr.window_should_close():
        
        if game_state == GAME_ACTIVE:
            update_game()
        elif game_state == GAME_OVER:
            # Check for restart command
            if pr.is_key_pressed(pr.KEY_R):
                initialize_game()
        
        draw_game()

    # De-Initialization
    pr.close_window()

if __name__ == "__main__":
    main()

