A program that simulates a human soul.

# Idea

Spilt the model into a base and memory.
Base interacts the memory to remember and process thoughts.
Memory is divided into reflex, shortTerm and longTerm.
Whenever model interacts with something with will store in shortTerm memory.
Then when ever the model is in idle then shortTerm memory transfer to longTerm only important part and most repeating part trasfer to reflex.
All habits in reflex have resistence in order to prevent chainging fast.
Habits in reflex only change when there is only number of repeations reached a saturation point.

# Implementations

## Game AI
Can fully be implement? Yes for decision
**Game Engine**: Godot
**Replica**: Nemesis System
**Organic Decision**: Yes
**Real Time**: Yes
**Limited**: Decision

## NLP
Can fully be implement? No for reflex
**LM**: RWKV-G1
**Replica**: J.A.R.V.I.S. , Great Sage
**Organic Decision**: No
**Limited**: RAG System and Decision
**Real Time**: No but depends on hardware. 