from enum import Enum, auto

class soul():
    def __init__(self):
        # Configuration
        self.saturationPoint = 3
        self.debugMode = True

        # Activating reflex variables
        self.reflexData = reflex(saturationPoint)
        self.shortTermMemory = []
        if debugMode:
            print("Soul started initialised her reflex")

    class reflex():
        def __init__(self, target):
            self.habitName = "Neutral"
            self.resistanceCount = 0
            self.saturationTarget = target

    # Interaction Phase
    def precieveEvent(self, eventType):
        self.shortTermMemory.append(eventType)
        if self.debugMode:
            print("Soul adding her moments")

    def currentHabit(self):
        return reflexData.habitName

    # Idle phase
    def idleState(self):
        if self.shortTermMemory is None:
            return
        if self.debugMode:
            print("Soul is process her short term memory")

        # Analyze frequent event
        counts = {}
        mostCommonEvent = ""
        highestCount = 0

        for event in self.shortTermMemory:
            if not event in counts:
                counts[event] = 0
                counts[event] += 1
                if counts[event] > highestCount:
                    highestCount = counts[event]
                    mostCommonEvent = event

            # Decide most optimal startegy
            desiredHabit = self.decideCounterStartegy(mostCommonEvent)

            # Updating reflex
            resistanceLogic(desiredHabit)

            # Wipe short term memory
            self.shortTermMemory.clear()

    def decideCounterStartegy(self, eventType):
        match eventType:
            case _:
                return "Normal"

    def resistanceLogic(self, newHabit):
        # Case 1: Refocing existing habit
        if newHabit == self.reflexData.habitName:
            self.reflexData.resistanceCount = 0
            if debugMode:
                print("Soul is renforcing her habit")
            return

        # Case 2: Try to change habit
        self.reflexData.resistanceCount += 1
        if debugMode:
            print("Soul wants to change her habit")
            print(f"Soul resistance:{self.reflexData.resistanceCount}, {self.reflexData.saturationTarget}")

        # Check saturation
        if self.reflexData.resistanceCount >= self.reflexData.saturationPoint:
            self.reflexData.habitName = newHabit
            self.reflexData.resistanceCount
            if debugMode:
                print("Soul changes her habit")

