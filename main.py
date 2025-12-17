import time
from collections import defaultdict

# ==============================================================================
# 1. MEMORY MODULE (FIXED: retrieve_reflex added)
# ==============================================================================

class MemoryModule:
    """Manages Short-Term (ST), Long-Term (LT), and Reflex (R) memory stores."""

    def __init__(self):
        # Configuration Constants (Tunable Hyperparameters)
        self.ST_CAPACITY = 3        # Reduced for quick demonstration
        self.IMPORTANCE_THRESHOLD = 0.5  # Adjusted for demo ease
        self.REPETITION_THRESHOLD = 2
        self.INITIAL_RESISTANCE = 3.0

        # Memory Stores
        self.short_term = []
        self.long_term = []
        self.reflex = {}

    # --- Interaction Methods ---
    def store_short_term(self, input_data, action, importance):
        """Stores a new interaction event into ST memory."""
        event = {'input': input_data, 'action': action, 'importance': importance, 'timestamp': time.time()}
        self.short_term.append(event)

    def retrieve_reflex(self, trigger):
        """Checks R for a matching habit."""
        return self.reflex.get(trigger) # <--- THE CRUCIAL FIX

    # --- Resistance and Saturation Logic ---
    def update_reflex_resistance(self, trigger, change, new_response=None):
        """Updates resistance and handles saturation (habit replacement)."""
        if trigger in self.reflex:
            current_resistance = self.reflex[trigger]['resistance_counter']

            # 1. Update Resistance
            self.reflex[trigger]['resistance_counter'] = max(0, current_resistance + change)

            # 2. Check for Saturation (Change < 0 indicates an override attempt)
            if self.reflex[trigger]['resistance_counter'] <= 0 and change < 0:
                print(f"\n!!! HABIT SATURATION REACHED !!!")
                print(f"    - Old Habit: '{trigger}' -> '{self.reflex[trigger]['response']}'")

                if new_response:
                    # Replace the habit
                    self.reflex[trigger]['response'] = new_response
                    self.reflex[trigger]['resistance_counter'] = self.INITIAL_RESISTANCE
                    print(f"    - New Habit Formed: '{trigger}' -> '{new_response}' (Resistance reset to {self.INITIAL_RESISTANCE})")
                else:
                    del self.reflex[trigger]
                    print(f"    - Habit deleted.")

            else:
                print(f"  [R Update] '{trigger}' resistance is now {self.reflex[trigger]['resistance_counter']:.1f}")

    # --- Consolidation Methods (Idle Processing) ---
    def run_consolidation_loop(self):
        """Executes the ST -> LT and ST/LT -> R transfers."""
        self._consolidate_st_to_lt()
        self._identify_and_form_reflexes()

    def _consolidate_st_to_lt(self):
        """Transfer important ST events to LT and clear ST."""
        newly_consolidated = []
        for event in self.short_term:
            if event['importance'] >= self.IMPORTANCE_THRESHOLD:
                concept = event['input']
                self.long_term.append({'concept': concept, 'source_action': event['action']})
                newly_consolidated.append(concept)

        print(f"\n--- CONSOLIDATION: ST -> LT ---")
        print(f"  Consolidated {len(newly_consolidated)} important events.")
        self.short_term.clear()

    def _identify_and_form_reflexes(self):
        """Scan LT for repeating patterns and form habits."""
        all_interactions = self.long_term

        pattern_counts = defaultdict(int)
        for interaction in all_interactions:
            key = (interaction['concept'], interaction['source_action'])
            pattern_counts[key] += 1

        new_reflexes = 0
        for (trigger, response), count in pattern_counts.items():
            if count >= self.REPETITION_THRESHOLD and trigger not in self.reflex:
                self.reflex[trigger] = {'response': response, 'resistance_counter': self.INITIAL_RESISTANCE}
                new_reflexes += 1
                print(f"  [R Formed] NEW HABIT: '{trigger}' -> '{response}' (Resistance: {self.INITIAL_RESISTANCE})")
        print(f"--- CONSOLIDATION: R Formation ---\n")


# ==============================================================================
# 2. BASE MODEL (Processor and Decision Maker)
# ==============================================================================

class BaseModel:
    """The processor that takes input, decides action, and manages memory interaction."""

    def __init__(self, memory_module):
        self.memory = memory_module

    def _calculate_novelty(self, input_data):
        is_familiar = any(input_data in event['concept'] for event in self.memory.long_term)
        return 0.9 if not is_familiar else 0.2

    def _decide_action(self, input_data, effort, user_feedback):
        novelty = self._calculate_novelty(input_data)

        # Importance simplified for the demo: Importance = Effort
        importance = effort

        reflex_found = self.memory.retrieve_reflex(input_data)
        action_output = f"New Processed Response: {input_data}" # Default action

        if reflex_found:
            reflex_response = reflex_found['response']

            # Override Condition: If Importance > 0.8
            if importance > 0.8:
                new_action = f"FORCED NEW ACTION: {input_data} (Importance:{importance:.2f})"
                action_output = new_action
                self.memory.update_reflex_resistance(input_data, change=-1.0, new_response=new_action)
            else:
                # Use Reflex: Reinforce the existing habit slightly
                self.memory.update_reflex_resistance(input_data, change=+0.1)
                action_output = f"REFLEX ACTION (Resistance:{reflex_found['resistance_counter']:.1f}): {reflex_response}"

        return action_output, importance

    def process_input(self, input_data, effort=0.5, user_feedback=0):
        action, importance = self._decide_action(input_data, effort, user_feedback)

        print(f"\n[INTERACTION] Input: '{input_data}' | Importance: {importance:.2f}")
        print(f"  -> Action Taken: {action}")

        self.memory.store_short_term(input_data, action, importance)

        if len(self.memory.short_term) >= self.memory.ST_CAPACITY:
            print(f"\n[SYSTEM] ST capacity reached ({self.memory.ST_CAPACITY}). Starting idle consolidation...")
            self.memory.run_consolidation_loop()

        return action

# ==============================================================================
# 3. DEMONSTRATION RUN
# ==============================================================================

if __name__ == "__main__":
    memory = MemoryModule()
    base = BaseModel(memory)
    habit_phrase = "Good morning"

    print("=========================================")
    print("PHASE 1: LEARNING & HABIT FORMATION")
    print("=========================================")

    # Events 1 & 2: Importance is 0.5 (>= THRESHOLD 0.5). They transfer to LT.
    base.process_input(habit_phrase, effort=0.5, user_feedback=0.0)
    base.process_input(habit_phrase, effort=0.5, user_feedback=0.0)

    # Event 3: Consolidation runs. LT now has 2 instances. Reflex is formed.
    base.process_input(habit_phrase, effort=0.5, user_feedback=0.0)

    print("=========================================")
    print("PHASE 2: HABIT IS ACTIVE AND REINFORCED")
    print("=========================================")

    # Event 4: Reflex is used. Resistance increases (3.0 -> 3.1).
    base.process_input(habit_phrase, effort=0.5, user_feedback=0.0)

    print("=========================================")
    print("PHASE 3: HIGH-IMPORTANCE OVERRIDE AND REPLACEMENT")
    print("=========================================")

    # Event 5 (Override 1): Importance 0.9. Resistance reduced by 1.0 (3.1 -> 2.1).
    base.process_input(habit_phrase, effort=0.9, user_feedback=0.0)

    # Event 6 (Override 2): Resistance reduced by 1.0 (2.1 -> 1.1).
    base.process_input(habit_phrase, effort=0.9, user_feedback=0.0)

    # Event 7 (Override 3): Resistance reduced by 1.0 (1.1 -> 0.1).
    base.process_input(habit_phrase, effort=0.9, user_feedback=0.0)

    # Event 8 (Override 4): Resistance reaches zero. Habit is replaced and reset to 3.0.
    base.process_input(habit_phrase, effort=0.9, user_feedback=0.0)

    print("\n=========================================")
    print("FINAL STATE CHECK: HABIT HAS BEEN REPLACED")
    print("=========================================")
    print(f"LT Size: {len(memory.long_term)}")
    print(f"Reflexes: {memory.reflex}")
