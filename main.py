import time
from collections import defaultdict

# ==============================================================================
# 1. MEMORY MODULE (Data Storage and Logic)
# ==============================================================================

class MemoryModule:
    """Manages ST, LT, and R, including memory strength/decay."""

    def __init__(self):
        # Configuration Constants
        self.ST_CAPACITY = 3
        self.IMPORTANCE_THRESHOLD = 0.5
        self.REPETITION_THRESHOLD = 2
        self.INITIAL_RESISTANCE = 3.0

        # Decay Constants
        self.DECAY_RATE_PER_CYCLE = 0.2
        self.MIN_STRENGTH_THRESHOLD = 0.4

        # Memory Stores
        self.short_term = []
        self.long_term = []
        self.reflex = {}

    # --- Interaction Methods ---
    def store_short_term(self, input_data, action, importance, affective_tag):
        """Stores a new interaction event, including the emotional tag."""
        event = {
            'input': input_data,
            'action': action,
            'importance': importance,
            'timestamp': time.time(),
            'affective_tag': affective_tag
        }
        self.short_term.append(event)

    def retrieve_reflex(self, trigger):
        """Checks R for a matching habit."""
        return self.reflex.get(trigger)

    # --- Resistance and Saturation Logic ---
    def update_reflex_resistance(self, trigger, change, new_response=None):
        """Updates resistance and handles saturation (habit replacement)."""
        if trigger in self.reflex:
            current_resistance = self.reflex[trigger]['resistance_counter']
            self.reflex[trigger]['resistance_counter'] = max(0, current_resistance + change)

            if self.reflex[trigger]['resistance_counter'] <= 0 and change < 0:
                print(f"\n!!! HABIT SATURATION REACHED !!!")
                print(f"    - Old Habit: '{trigger}' -> '{self.reflex[trigger]['response']}'")

                if new_response:
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
        """Now includes the decay step first."""
        self._apply_memory_decay()
        self._consolidate_st_to_lt()
        self._identify_and_form_reflexes()

    def _apply_memory_decay(self):
        """Reduces the strength of LT items and purges forgotten ones."""
        memories_to_keep = []
        forgotten_count = 0

        for memory in self.long_term:
            memory['strength'] -= self.DECAY_RATE_PER_CYCLE

            if memory['strength'] > self.MIN_STRENGTH_THRESHOLD:
                memories_to_keep.append(memory)
            else:
                forgotten_count += 1

        self.long_term = memories_to_keep
        print(f"\n--- DECAY: Forgetting Curve Applied ---")
        print(f"  Purged {forgotten_count} LT memories due to low strength.")

    def _consolidate_st_to_lt(self):
        """Transfer important ST events to LT and clear ST."""
        newly_consolidated = []
        for event in self.short_term:
            if event['importance'] >= self.IMPORTANCE_THRESHOLD:
                self.long_term.append({
                    'concept': event['input'],
                    'source_action': event['action'],
                    'strength': 1.0,
                    'affective_tag': event['affective_tag']
                })
                newly_consolidated.append(event['input'])

        print(f"--- CONSOLIDATION: ST -> LT ---")
        print(f"  Consolidated {len(newly_consolidated)} important events.")
        self.short_term.clear()

    def _identify_and_form_reflexes(self):
        """Scan LT for repeating patterns and form habits."""
        all_interactions = self.long_term
        pattern_counts = defaultdict(int)

        # Only count patterns from LT memories that have significant strength (> 0.5)
        strong_memories = [m for m in all_interactions if m['strength'] > 0.5]

        for interaction in strong_memories:
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
    """The processor that tracks emotional state and calculates importance."""

    def __init__(self, memory_module):
        self.memory = memory_module
        self.emotional_state = 0.0 # Initial state is neutral

    def _calculate_novelty(self, input_data):
        is_familiar = any(input_data in event['concept'] for event in self.memory.long_term)
        return 0.9 if not is_familiar else 0.2

    def _update_emotional_state(self, user_feedback):
        """Adjusts emotional state based on input feedback."""
        self.emotional_state = max(-1.0, min(1.0, self.emotional_state + (user_feedback * 0.3)))
        return self.emotional_state

    def _decide_action(self, input_data, effort, user_feedback):

        affective_tag = self._update_emotional_state(user_feedback)
        emotional_magnitude = abs(affective_tag)

        # Importance calculation (weighted heavily by emotional magnitude)
        novelty = self._calculate_novelty(input_data)
        importance = (novelty * 0.2) + (effort * 0.2) + (emotional_magnitude * 0.6)

        reflex_found = self.memory.retrieve_reflex(input_data)
        action_output = f"New Processed Response: {input_data}"

        if reflex_found:
            reflex_response = reflex_found['response']

            # Override Condition: If Importance > 0.8
            if importance > 0.8:
                new_action = f"FORCED NEW ACTION: {input_data} (Emotionally Urgent: {affective_tag:.2f})"
                action_output = new_action
                self.memory.update_reflex_resistance(input_data, change=-1.0, new_response=new_action)
            else:
                self.memory.update_reflex_resistance(input_data, change=+0.1)
                action_output = f"REFLEX ACTION (Resistance:{reflex_found['resistance_counter']:.1f}): {reflex_response}"

        return action_output, importance, affective_tag

    def process_input(self, input_data, effort=0.5, user_feedback=0):
        action, importance, affective_tag = self._decide_action(input_data, effort, user_feedback)

        print(f"\n[INTERACTION] Input: '{input_data}' | Importance: {importance:.2f} | Emotion: {affective_tag:.2f}")
        print(f"  -> Action Taken: {action}")

        self.memory.store_short_term(input_data, action, importance, affective_tag)

        if len(self.memory.short_term) >= self.memory.ST_CAPACITY: # FIXED: Removed nested 'memory'
            print(f"\n[SYSTEM] ST capacity reached ({self.memory.ST_CAPACITY}). Starting idle consolidation...") # FIXED: Removed nested 'memory'
            self.memory.run_consolidation_loop()

        return action

# ==============================================================================
# 3. DEMONSTRATION RUN
# ==============================================================================

if __name__ == "__main__":
    memory = MemoryModule()
    base = BaseModel(memory)
    habit_phrase = "Good morning"
    urgent_phrase = "Must remember this"

    print("=========================================")
    print("PHASE 1: EMOTIONAL LEARNING & HABIT FORMATION")
    print("=========================================")

    # Events 1-3: Low effort/emotion. Consolidation runs after event 3. Reflex is formed.
    base.process_input(habit_phrase, effort=0.4, user_feedback=0.0)
    base.process_input(habit_phrase, effort=0.4, user_feedback=0.0)
    base.process_input(habit_phrase, effort=0.4, user_feedback=0.0)

    print("=========================================")
    print("PHASE 2: DECAY & REPLACEMENT")
    print("=========================================")

    # Event 4: Use Habit (Reinforce)
    base.process_input(habit_phrase, effort=0.4, user_feedback=0.0)

    # Event 5: High NEGATIVE emotion interaction (Increases importance and lowers emotional state).
    base.process_input(urgent_phrase, effort=0.4, user_feedback=-1.0)

    # Event 6: Consolidate. The DECAY logic runs first, reducing the strength of LT memories.
    # The urgent memory is stored in LT with high initial importance.
    base.process_input("Random input", effort=0.4, user_feedback=0.0)

    print("=========================================")
    print("PHASE 3: HABIT REPLACEMENT (Forced Override)")
    print("=========================================")

    # Use high effort/novelty/emotion to force overrides until replacement occurs.
    base.process_input(habit_phrase, effort=0.9, user_feedback=0.0)
    base.process_input(habit_phrase, effort=0.9, user_feedback=0.0)
    base.process_input(habit_phrase, effort=0.9, user_feedback=0.0)
    base.process_input(habit_phrase, effort=0.9, user_feedback=0.0)

    print("\n=========================================")
    print("FINAL STATE CHECK: LT Decay and Reflex Replacement")
    print("=========================================")

    # Run two more consolidation cycles without any input to force further decay (ST is empty)
    print("\n[SYSTEM] Forcing two decay cycles...")
    base.memory.run_consolidation_loop()
    base.memory.run_consolidation_loop()


    print(f"\nLT Size: {len(base.memory.long_term)} (Check if low-strength memories were purged)")
    print("--- LONG TERM MEMORIES ---")
    for mem in base.memory.long_term:
        print(f"  Input: {mem['concept'][:20]}... | Strength: {mem['strength']:.2f} | Emotion: {mem['affective_tag']:.2f}")

    print("\n--- REFLEXES ---")
    print(f"Reflexes: {base.memory.reflex}")
