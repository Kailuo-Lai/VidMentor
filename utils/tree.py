class TreeNode:
    def __init__(self, title, description=''):
        self.title = title
        self.description = description
        self.children = []
        self.refer_time = []
        self.key_words = []

    def add_child(self, child_node):
        self.children.append(child_node)

    def __repr__(self, level=0):
        ret = "\t" * level + repr(self.title) + " - " + repr(self.description) + "\n"
        for child in self.children:
            ret += child.__repr__(level + 1)
        return ret
    
def parse_markdown_to_tree(markdown_str,tree_name='root'):
    lines = markdown_str.strip().split('\n')
    root = TreeNode(tree_name)
    stack = [(root, 0)]
    current_node = root
    current_level = 0

    for i, line in enumerate(lines):
        stripped_line = line.lstrip()
        if len(stripped_line)==0:
            continue
        if stripped_line.startswith('#'):
            level = stripped_line.count('#')
            title = stripped_line.lstrip('#').strip()
            description = ''
            current_level = level
            
        elif stripped_line.startswith('-'):
            level = current_level + 1
            title = stripped_line.lstrip('-').strip()
            description = ''
        else:
            if current_node.title != "root":
                current_node.description = stripped_line.strip()
            continue

        node = TreeNode(title, description)
        
        while stack and stack[-1][1] >= level:
            stack.pop()

        stack[-1][0].add_child(node)
        stack.append((node, level))
        current_node = node

    return root

def get_leaf_node_paths(node, path=""):
    # Initialize the path if it's the first call
    if path == "":
        path = f"{node.title}:{node.description}"

    # If the node is a leaf, return the path
    if not node.children:
        return [path]

    # Otherwise, extend the path and continue the search in the children
    leaf_paths = []
    for child in node.children:
        child_path = f"{path}->{child.title}:{child.description}"
        leaf_paths.extend(get_leaf_node_paths(child, child_path))
    return leaf_paths