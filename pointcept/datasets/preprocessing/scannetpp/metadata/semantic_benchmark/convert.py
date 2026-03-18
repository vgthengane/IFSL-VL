def write_to_new_file(txt_file, new_file):
    """Writes each line from the input txt file to a new file with double quotation marks.

    Args:
      txt_file: The path to the input txt file.
      new_file: The path to the output file.
    """

    with open(txt_file, "r") as f:
        lines = f.readlines()

    with open(new_file, "w") as f:
        for line in lines:
            f.write('"')
            f.write(line.strip())
            f.write('",\n')


def save_dict_keys_to_file(dictionary, filename):
    """Saves the keys of a dictionary to a file.

    Args:
      dictionary: The dictionary to save.
      filename: The name of the output file.
    """

    with open(filename, "w") as f:
        for key in dictionary.keys():
            f.write(f'"{key}",\n')


# Example usage:
# input_file = "top100.txt"
# output_file = "new.py"
# write_to_new_file(input_file, output_file)
save_dict_keys_to_file(
    {
        "wall": 280,
        "floor": 279,
        "door": 271,
        "ceiling": 254,
        "table": 207,
        "window": 188,
        "box": 179,
        "ceiling lamp": 177,
        "light switch": 177,
        "cabinet": 174,
        "chair": 171,
        "heater": 171,
        "monitor": 143,
        "whiteboard": 134,
        "office chair": 133,
        "bottle": 131,
        "doorframe": 130,
        "keyboard": 124,
        "window frame": 123,
        "mouse": 112,
        "paper": 104,
        "blinds": 100,
        "trash can": 99,
        "telephone": 99,
        "book": 95,
        "shelf": 91,
        "sink": 88,
        "windowsill": 83,
        "bag": 82,
        "smoke detector": 82,
        "storage cabinet": 76,
        "electrical duct": 74,
        "bookshelf": 70,
        "towel": 68,
        "backpack": 62,
        "cup": 61,
        "curtain": 60,
        "pipe": 57,
        "computer tower": 57,
        "plant": 55,
        "picture": 55,
        "pillow": 54,
        "power strip": 54,
        "laptop": 53,
        "jacket": 52,
        "whiteboard eraser": 52,
    },
    "new.txt",
)
