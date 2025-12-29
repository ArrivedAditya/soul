name = "SOUL"
print(f"{name} started.")
for i in range(100):
    value = input("> ")
    match value:
        case ":q": break
    print(f"{name}: ------------------------------------")


print(f"{name} ended.")
