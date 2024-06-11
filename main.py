import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import random

# Load data from CSV file
tasks = pd.read_csv('tasks.csv')

# Debug: Print the original data
print("Original Data:")
print(tasks)

# Fill NaNs with empty strings if necessary
tasks['description'].fillna('', inplace=True)

# Strip whitespace from descriptions and filter out empty descriptions
tasks['description'] = tasks['description'].str.strip()
tasks = tasks[tasks['description'] != '']

# Debug: Print the filtered data
print("\nFiltered Data:")
print(tasks)

# Ensure there are no empty descriptions and all strings are valid
if tasks['description'].str.len().sum() == 0:
    raise ValueError("All descriptions are either empty or invalid.")

# Define the pipeline with adjusted CountVectorizer parameters
model = Pipeline([
    ('vectorizer', CountVectorizer(stop_words=None, lowercase=True)),
    ('classifier', LogisticRegression())
])

# Fit the model
model.fit(tasks['description'], tasks['priority'])

# Debug: Print a success message
print("\nModel fitted successfully.")

# Function to save tasks to CSV
def save_tasks():
    tasks.to_csv('tasks.csv', index=False)

# Function to add a task to the list
def add_task(description, priority):
    global tasks  # Declare tasks as a global variable
    new_task = pd.DataFrame({'description': [description], 'priority': [priority]})
    tasks = pd.concat([tasks, new_task], ignore_index=True)
    save_tasks()

# Function to remove a task by description
def remove_task(description):
    global tasks  # Declare tasks as a global variable
    tasks = tasks[tasks['description'] != description]
    save_tasks()

# Function to list all tasks
def list_tasks():
    if tasks.empty:
        print("No tasks available.")
    else:
        print(tasks)

# Function to recommend a task based on machine learning
def recommend_task():
    if not tasks.empty:
        # Get high-priority tasks
        high_priority_tasks = tasks[tasks['priority'] == 'High']
        
        if not high_priority_tasks.empty:
            # Choose a random high-priority task
            random_task = random.choice(high_priority_tasks['description'].tolist())
            print(f"Recommended task: {random_task} - Priority: High")
        else:
            print("No high-priority tasks available for recommendation.")
    else:
        print("No tasks available for recommendations.")

# Main menu
while True:
    print("\nTask Management App")
    print("1. Add Task")
    print("2. Remove Task")
    print("3. List Tasks")
    print("4. Recommend Task")
    print("5. Exit")

    choice = input("Select an option: ")

    if choice == "1":
        description = input("Enter task description: ")
        priority = input("Enter task priority (Low/Medium/High): ").capitalize()
        add_task(description, priority)
        print("Task added successfully.")

    elif choice == "2":
        description = input("Enter task description to remove: ")
        remove_task(description)
        print("Task removed successfully.")

    elif choice == "3":
        list_tasks()

    elif choice == "4":
        recommend_task()

    elif choice == "5":
        print("Goodbye!")
        break

    else:
        print("Invalid option. Please select a valid option.")
