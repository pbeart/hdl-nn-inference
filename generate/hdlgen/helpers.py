def indent(text, amount):
    return "\n".join([line if len(line) == 0 else amount * "  " + line for line in text.split("\n")])

indentation = "  "
