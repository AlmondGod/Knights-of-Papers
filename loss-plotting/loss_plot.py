import matplotlib.pyplot as plt

def plot_loss(file_path):
    # Read the file contents
    with open(file_path, 'r') as file:
        losses = [float(line.strip()) for line in file if line.strip()]

    # Create episode numbers (assuming each line is a step)
    episodes = list(range(1, len(losses) + 1))

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, losses)
    plt.title('Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.ylim(bottom=0)

    # Show the plot
    plt.show()

# Usage
file_path = 'actorlossessac.txt'  # Replace with your actual file path
plot_loss(file_path)