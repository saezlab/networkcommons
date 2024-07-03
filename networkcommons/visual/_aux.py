
def wrap_node_name(node_name):
    if ":" in node_name:
        node_name = node_name.replace(":", "_")
    if node_name.startswith("COMPLEX"):
        # remove the word COMPLEX with a separator (:/-, etc)
        return node_name[8:]
    else:
        return node_name


