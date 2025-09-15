import random

class_colors = {}

def get_class_color(class_name: str):
    """Assign unique, consistent color for each class."""
    if class_name not in class_colors:
        class_colors[class_name] = (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255),
        )
    return class_colors[class_name]


