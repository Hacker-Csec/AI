import customtkinter as ctk
import json
import os
from pathlib import Path

class TrainingChatbox(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Training Data Editor")
        self.geometry("1000x800")
        
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Create main frame
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)
        
        # Create scrollable frame for training examples
        self.scrollable_frame = ctk.CTkScrollableFrame(self.main_frame)
        self.scrollable_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.scrollable_frame.grid_columnconfigure(0, weight=1)
        
        # Create input frame at the bottom
        self.input_frame = ctk.CTkFrame(self.main_frame)
        self.input_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        self.input_frame.grid_columnconfigure(0, weight=1)
        
        # Create input fields
        self.instruction_label = ctk.CTkLabel(self.input_frame, text="Instruction:")
        self.instruction_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        self.instruction_entry = ctk.CTkEntry(self.input_frame, width=400)
        self.instruction_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        self.input_label = ctk.CTkLabel(self.input_frame, text="Input:")
        self.input_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        
        self.input_entry = ctk.CTkEntry(self.input_frame, width=400)
        self.input_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        
        self.output_label = ctk.CTkLabel(self.input_frame, text="Output:")
        self.output_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")
        
        self.output_entry = ctk.CTkEntry(self.input_frame, width=400)
        self.output_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        
        # Create buttons
        self.add_button = ctk.CTkButton(
            self.input_frame,
            text="Add Example",
            command=self.add_example
        )
        self.add_button.grid(row=3, column=0, padx=5, pady=10)
        
        self.save_button = ctk.CTkButton(
            self.input_frame,
            text="Save Training Data",
            command=self.save_training_data
        )
        self.save_button.grid(row=3, column=1, padx=5, pady=10)
        
        # Create status label
        self.status_label = ctk.CTkLabel(self.input_frame, text="")
        self.status_label.grid(row=4, column=0, columnspan=2, padx=5, pady=5)
        
        # Initialize training data
        self.training_data = []
        self.load_training_data()
        
    def load_training_data(self):
        """Load existing training data from JSON file."""
        data_dir = Path("training_data")
        data_dir.mkdir(exist_ok=True)
        
        training_file = data_dir / "training_data.json"
        if training_file.exists():
            try:
                with open(training_file, 'r', encoding='utf-8') as f:
                    self.training_data = json.load(f)
                self.refresh_examples()
                self.status_label.configure(text=f"Loaded {len(self.training_data)} training examples")
            except Exception as e:
                self.status_label.configure(text=f"Error loading training data: {e}")
    
    def refresh_examples(self):
        """Refresh the display of training examples."""
        # Clear existing examples
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        # Add each example
        for i, example in enumerate(self.training_data):
            example_frame = ctk.CTkFrame(self.scrollable_frame)
            example_frame.grid(row=i, column=0, padx=5, pady=5, sticky="ew")
            example_frame.grid_columnconfigure(0, weight=1)
            
            # Create labels for the example
            instruction_text = f"Instruction: {example['instruction']}"
            input_text = f"Input: {example['input']}"
            output_text = f"Output: {example['output']}"
            
            ctk.CTkLabel(example_frame, text=instruction_text, wraplength=800).grid(row=0, column=0, padx=5, pady=2, sticky="w")
            ctk.CTkLabel(example_frame, text=input_text, wraplength=800).grid(row=1, column=0, padx=5, pady=2, sticky="w")
            ctk.CTkLabel(example_frame, text=output_text, wraplength=800).grid(row=2, column=0, padx=5, pady=2, sticky="w")
            
            # Add delete button
            delete_button = ctk.CTkButton(
                example_frame,
                text="Delete",
                command=lambda idx=i: self.delete_example(idx)
            )
            delete_button.grid(row=3, column=0, padx=5, pady=5, sticky="e")
    
    def add_example(self):
        """Add a new training example."""
        instruction = self.instruction_entry.get().strip()
        input_text = self.input_entry.get().strip()
        output = self.output_entry.get().strip()
        
        if instruction and input_text and output:
            example = {
                "instruction": instruction,
                "input": input_text,
                "output": output
            }
            
            self.training_data.append(example)
            self.refresh_examples()
            
            # Clear input fields
            self.instruction_entry.delete(0, "end")
            self.input_entry.delete(0, "end")
            self.output_entry.delete(0, "end")
            
            self.status_label.configure(text=f"Added new example. Total examples: {len(self.training_data)}")
    
    def delete_example(self, index):
        """Delete a training example."""
        if 0 <= index < len(self.training_data):
            self.training_data.pop(index)
            self.refresh_examples()
            self.status_label.configure(text=f"Deleted example. Total examples: {len(self.training_data)}")
    
    def save_training_data(self):
        """Save training data to JSON file."""
        data_dir = Path("training_data")
        data_dir.mkdir(exist_ok=True)
        
        training_file = data_dir / "training_data.json"
        try:
            with open(training_file, 'w', encoding='utf-8') as f:
                json.dump(self.training_data, f, indent=4)
            self.status_label.configure(text=f"Training data saved successfully! {len(self.training_data)} examples saved.")
            print("Training data saved successfully!")
        except Exception as e:
            self.status_label.configure(text=f"Error saving training data: {e}")
            print(f"Error saving training data: {e}")

if __name__ == "__main__":
    app = TrainingChatbox()
    app.mainloop() 