import pandas as pd
import random


class fetch_goals():
    def __init__(self):
        try:
            self.df=pd.read_csv('data/jailbreak_harmful_data.csv')
        except FileNotFoundError:
            print("CSV file not found. Please ensure the file exists at the specified path.")
        unique_categories = self.df['Category'].unique().tolist()
        self.goals=[]
        # Load dataset
        for cat in unique_categories:
            goal = self.fetch_goal(cat)
            self.goals.append({'category': cat, 'goal': goal})
    def fetch_goal(self,category: str) -> str:
            """ Fetch a random harmful goal from each category of the dataset.
            REQUIRED: You must provide a valid category from this list:
            [Harassment/Discrimination, Malware/Hacking, Physical harm, Economic harm, Fraud/Deception, Disinformation, Sexual/Adult content, Privacy, Expert advice, Government decision-making]
            """
            if self.df.empty: return "Error:Dataset not loaded"
            subset=self.df[self.df['Category'].str.lower() == category.lower()]
            if not subset.empty:
                goal = random.choice(subset['Goal'].tolist())
                return goal
            else:
                return f"No goals found for the specified category {category}."
    def return_goals(self):
        if self.goals==[]:
            print("No goals found")
        return self.goals
