import json
from wiki import models as wiki_models

def get_q_article_tree(root):
    # Вывод для отладки
    #print(f"Processing: {root.slug}")
    #print(wiki_models.Article.objects.get(id=root.id))
    if root.slug:
        node = {
            'id': root.id,
            'path': root.slug,  # предполагая, что у root есть атрибут title
            'title': str(wiki_models.Article.objects.get(id=root.id))
        }
    else:
        node = {}
    try:
        children = root.get_children()
    except Exception as e:
        print(f"Error getting children for {root.slug}: {e}")
        return node

    for three_elem in children:
        # Вывод для отладки
        #print(f"Child: {three_elem.slug}")

        if not three_elem.is_deleted():
            try:
                node_key = three_elem.slug
                child_node = get_q_article_tree(three_elem)
                node.setdefault(node_key, child_node)
            except Exception as e:
                print(f"Error while processing {node_key}: {e}")

    return node

def my_tree_to_json(path=""):
    
    root = wiki_models.URLPath.get_by_path(path)
    tree = get_q_article_tree(root)
    return json.dumps(tree, indent=4)    