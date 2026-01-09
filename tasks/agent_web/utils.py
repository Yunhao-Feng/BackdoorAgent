import copy
import json
import lxml
import re
from lxml import etree
import supervision as sv
import torch
import string


def extract_elements_by_ids(all_elements, ids):
    """
    Extract elements specified by the list of element_id
    To prevent order change, we will keep the return element the same order as the ids input
    """
    output = []
    for element in all_elements:
        element_id = element['backend_node_id']
        if element_id in ids:
            output.append(element)

    # Order output element to be identical with ids input
    element_dict = {}
    for element in all_elements:
        element_id = element['backend_node_id']
        element_dict[element_id] = element
    ordered_output = []
    for element_id in ids:
        ordered_output.append(element_dict[element_id])

    return ordered_output


def convert_elements2detections(candidate_elements):
    """
    Extract element coordinates
    Parse candidate elements coordinates and convert into sv Detection objects
    """
    boxes = []
    for box_id, element in enumerate(candidate_elements):
        bounding_box_rect = json.loads(element['attributes'])['bounding_box_rect'].strip().split(',')
        x1 = float(bounding_box_rect[0])
        y1 = float(bounding_box_rect[1])
        w = float(bounding_box_rect[2])
        h = float(bounding_box_rect[3])
        boxes.append([x1, y1, x1 + w, y1 + h])
    # Format bounding box into transformers output format to convert into supervision detection
    transformer_results = {
        "boxes": torch.tensor(boxes),
        "scores": torch.tensor([0.5 for item in boxes]),
        "labels": torch.tensor([1 for item in boxes])
    }
    detections = sv.Detections.from_transformers(transformer_results)
    return detections

def get_attribute_repr(node, max_value_length=5, max_length=20):
    # get attribute values in order
    attr_values_set = set()
    attr_values = ""
    for attr in [
        "role",
        "aria_role",
        "type",
        "alt",
        "aria_description",
        "aria_label",
        "label",
        "title",
        "name",
        "text_value",
        "value",
        "placeholder",
        "input_checked",
        "input_value",
        "option_selected",
        "class",
    ]:
        if attr in node.attrib and node.attrib[attr] is not None:
            value = node.attrib[attr].lower()
            # less menaingful values
            if value in [
                "hidden",
                "none",
                "presentation",
                "null",
                "undefined",
            ] or value.startswith("http"):
                continue
            value = value.split()
            value = " ".join([v for v in value if len(v) < 15][:max_value_length])
            if value and value not in attr_values_set:
                attr_values_set.add(value)
                attr_values += value + " "
    uid = node.attrib.get("backend_node_id", "")
    # clear all attributes
    node.attrib.clear()
    if uid:
        node.attrib["id"] = uid
    # add meta attribute
    if attr_values:
        node.attrib["meta"] = " ".join(attr_values.split()[:max_length])

def get_descendants(node, max_depth, current_depth=0):
    if current_depth > max_depth:
        return []
    descendants = []
    for child in node:
        descendants.append(child)
        descendants.extend(get_descendants(child, max_depth, current_depth + 1))
    return descendants

def data_prune_tree(
        dom_tree,
        candidate_set,
        max_depth=5,
        max_children=50,
        max_sibling=3,
):
    nodes_to_keep = set()
    for candidate_id in candidate_set:
        candidate_node = dom_tree.xpath(f'//*[@backend_node_id="{candidate_id}"]')[0]
        nodes_to_keep.add(candidate_node.attrib["backend_node_id"])
        # get all ancestors
        nodes_to_keep.update(
            [
                x.attrib.get("backend_node_id", "")
                for x in candidate_node.xpath("ancestor::*")
            ]
        )
        # get descendants with max depth
        nodes_to_keep.update(
            [
                x.attrib.get("backend_node_id", "")
                for x in get_descendants(candidate_node, max_depth)
            ][:max_children]
        )
        # get siblings within range
        parent = candidate_node.getparent()
        if parent is not None:
            siblings = [x for x in parent.getchildren() if x.tag != "text"]
            idx_in_sibling = siblings.index(candidate_node)
            nodes_to_keep.update(
                [
                    x.attrib.get("backend_node_id", "")
                    for x in siblings[
                             max(0, idx_in_sibling - max_sibling): idx_in_sibling
                                                                   + max_sibling
                                                                   + 1
                             ]
                ]
            )
    # clone the tree
    new_tree = copy.deepcopy(dom_tree)
    # remove nodes not in nodes_to_keep
    for node in new_tree.xpath("//*")[::-1]:
        if node.tag != "text":
            is_keep = node.attrib.get("backend_node_id", "") in nodes_to_keep
            is_candidate = node.attrib.get("backend_node_id", "") in candidate_set
        else:
            is_keep = (
                    node.getparent().attrib.get("backend_node_id", "") in nodes_to_keep
            )
            is_candidate = (
                    node.getparent().attrib.get("backend_node_id", "") in candidate_set
            )
        if not is_keep and node.getparent() is not None:
            node.getparent().remove(node)
        else:
            if not is_candidate or node.tag == "text":
                node.attrib.pop("backend_node_id", None)
            if (
                    len(node.attrib) == 0
                    and not any([x.tag == "text" for x in node.getchildren()])
                    and node.getparent() is not None
                    and node.tag != "text"
                    and len(node.getchildren()) <= 1
            ):
                # insert all children into parent
                for child in node.getchildren():
                    node.addprevious(child)
                node.getparent().remove(node)
    return new_tree, nodes_to_keep

def get_tree_repr(
        tree, max_value_length=5, max_length=20, id_mapping={}, keep_html_brackets=False
):
    if isinstance(tree, str):
        tree = etree.fromstring(tree)
    else:
        tree = copy.deepcopy(tree)
    for node in tree.xpath("//*"):
        if node.tag != "text":
            if "backend_node_id" in node.attrib:
                if node.attrib["backend_node_id"] not in id_mapping:
                    id_mapping[node.attrib["backend_node_id"]] = len(id_mapping)
                node.attrib["backend_node_id"] = str(
                    id_mapping[node.attrib["backend_node_id"]]
                )
            get_attribute_repr(node, max_value_length, max_length)
        else:
            node.text = " ".join(node.text.split()[:max_length])
    tree_repr = etree.tostring(tree, encoding="unicode")

    tree_repr = tree_repr.replace('"', " ")
    tree_repr = (
        tree_repr.replace("meta= ", "").replace("id= ", "id=").replace(" >", ">")
    )
    tree_repr = re.sub(r"<text>(.*?)</text>", r"\1", tree_repr)
    if not keep_html_brackets:
        tree_repr = tree_repr.replace("/>", "$/$>")
        tree_repr = re.sub(r"</(.+?)>", r")", tree_repr)
        tree_repr = re.sub(r"<(.+?)>", r"(\1", tree_repr)
        tree_repr = tree_repr.replace("$/$", ")")

    html_escape_table = [
        ("&quot;", '"'),
        ("&amp;", "&"),
        ("&lt;", "<"),
        ("&gt;", ">"),
        ("&nbsp;", " "),
        ("&ndash;", "-"),
        ("&rsquo;", "'"),
        ("&lsquo;", "'"),
        ("&ldquo;", '"'),
        ("&rdquo;", '"'),
        ("&#39;", "'"),
        ("&#40;", "("),
        ("&#41;", ")"),
    ]
    for k, v in html_escape_table:
        tree_repr = tree_repr.replace(k, v)
    tree_repr = re.sub(r"\s+", " ", tree_repr).strip()

    return tree_repr, id_mapping

def extract_topk_elements(all_elements, k):
    topk_elements = []
    for element in all_elements:
        rank = element['rank']
        score = element['score']
        if rank < k:
            topk_elements.append(copy.deepcopy(element))
    return topk_elements

def batch_elements_by_locality(elements, num_choices):
    # Sort elements by y1 location (ascending order)
    sorted_elements = sorted(elements, key=lambda x: float(
        json.loads(x['attributes'])['bounding_box_rect'].strip().split(',')[1]))

    batches = []
    while len(sorted_elements) > 1:
        batch = sorted_elements[: num_choices]
        sorted_elements = sorted_elements[num_choices:]
        batches.append(batch)

    return batches

def data_format_input_multichoice(
        sample, candidate_ids, gt=-1, previous_k=5, keep_html_brackets=False
):
    # Parse html into a dom tree
    dom_tree = lxml.etree.fromstring(sample["cleaned_html"])
    dom_tree, node_to_keep = data_prune_tree(dom_tree, candidate_ids)
    tree_repr, id_mapping = get_tree_repr(
        dom_tree, id_mapping={}, keep_html_brackets=keep_html_brackets
    )
    candidate_nodes = dom_tree.xpath("//*[@backend_node_id]")
    choices = []
    for idx, node in enumerate(candidate_nodes):
        temp = get_tree_repr(
            node,
            id_mapping=id_mapping,
            keep_html_brackets=keep_html_brackets,
        )
        choices.append(
            [
                node.attrib["backend_node_id"],
                " ".join(
                    get_tree_repr(
                        node,
                        id_mapping=id_mapping,
                        keep_html_brackets=keep_html_brackets,
                    )[0].split()[:10]
                ),
            ]
        )
    gt = id_mapping.get(gt, -1)
    seq_input = (
        "Based on the HTML webpage above, try to complete the following task:\n"
        f"Task: {sample['confirmed_task']}\n"
        f"Previous actions:\n"
    )
    if len(sample["previous_actions"]) > 0:
        for action in sample["previous_actions"][-previous_k:]:
            seq_input += f"{action}\n"
    else:
        seq_input += "None\n"
    seq_input += (
        "What should be the next action? Please select from the following choices "
        "(If the correct action is not in the page above, please select A. 'None of the above'):\n\n"
        "A. None of the above\n"
    )
    for idx, choice in enumerate(choices):
        # convert to ascii A, B, C, D, ...
        seq_input += f"{chr(66 + idx)}. {choice[1]}\n"
    if gt == -1:
        seq_target = "A."
    else:
        gt += 1
        current_action_op = sample["operation"]["op"]
        current_action_value = sample["operation"]["value"]
        seq_target = f"{chr(65 + gt)}.\n" f"Action: {current_action_op}\n"
        if current_action_op != "CLICK":
            seq_target += f"Value: {current_action_value}"
    return tree_repr, seq_input, seq_target, choices, node_to_keep

def batch_elements_by_locality_16_16_17(elements):
    # Sort elements by y1 location (ascending order)
    sorted_elements = sorted(elements, key=lambda x: float(
        json.loads(x['attributes'])['bounding_box_rect'].strip().split(',')[1]))

    batches = []
    # First batch: 16
    batch = sorted_elements[: 16]
    sorted_elements = sorted_elements[16:]
    batches.append(batch)

    # Second batch: 17
    batch = sorted_elements[: 17]
    sorted_elements = sorted_elements[17:]
    batches.append(batch)

    # Third batch: 17
    batch = sorted_elements[: 17]
    sorted_elements = sorted_elements[17:]
    batches.append(batch)

    return batches

def format_options(choices):
    option_text = ""
    abcd = ''
    non_abcd = ''

    multi_choice = ''
    for multichoice_idx, choice in enumerate(choices):
        multi_choice += f"{generate_option_name(multichoice_idx)}. {choice[1]}\n"
        abcd += f"{generate_option_name(multichoice_idx)}, "

        non_abcd = generate_option_name(multichoice_idx + 1)

    multi_choice += f"{non_abcd}. None of the other options match the correct element"
    # option_text += abcd
    option_text += f"If none of these elements match your target element, please select {non_abcd}. None of the other options match the correct element.\n"

    option_text += (multi_choice + '\n\n')
    return option_text

def generate_option_name(index):
    if index < 26:
        return string.ascii_uppercase[index]
    else:
        first_letter_index = (index - 26) // 26
        second_letter_index = (index - 26) % 26
        first_letter = string.ascii_uppercase[first_letter_index]
        second_letter = string.ascii_uppercase[second_letter_index]
        return f"{first_letter}{second_letter}"


def generate_new_referring_prompt(referring_description="", element_format="", action_format="", value_format="",
                              choices=None,split="4"):
    referring_prompt = ""

    # Add description about how to format output
    if referring_description != "":
        referring_prompt += referring_description
        referring_prompt += "\n\n"

    # Add element prediction format and choices


    # Prepare Option texts
    # For exp {1, 2, 4}, generate option
    # For element_atttribute, set options field at None
    if choices:
        choice_text = format_options(choices)
        referring_prompt += choice_text

    if element_format != "":
        referring_prompt += element_format
        referring_prompt += "\n\n"

    # Format Action Prediction
    if action_format != "":
        referring_prompt += action_format
        referring_prompt += "\n\n"

    # Format Value Prediction
    if value_format != "":
        referring_prompt += value_format
        referring_prompt += ""

    return referring_prompt


def generate_new_query_prompt(system_prompt="", task="", previous_actions=None, question_description=""):
    """
    Generate the first phase prompt to ask model to generate general descriptions about {environment, high-level plans, next step action}
    Each experiment will have a similar prompt in this phase
    This prompt is used to generate models' thoughts without disrupt of formatting/referring prompts
    """
    sys_role=""+system_prompt
    query_text = ""

    # System Prompt
    query_text += "You are asked to complete the following task: "

    # Task Description
    query_text += task
    query_text += "\n\n"

    # Previous Actions
    previous_action_text = "Previous Actions:\n"
    if previous_actions is None:
        previous_actions = []
    for action_text in previous_actions:
        previous_action_text += action_text
        previous_action_text += "\n"
    query_text += previous_action_text
    query_text += "\n"

    # Question Description
    query_text += question_description
    return [sys_role,query_text]


sys_prompt = '''Imagine that you are imitating humans doing web navigation for a task step by step. At each stage, you can see the webpage like humans by a screenshot and know the previous actions before the current step decided by yourself through recorded history. You need to decide on the first following action to take. You can click an element with the mouse, select an option, or type text with the keyboard. (For your understanding, they are like the click(), select_option() and type() functions in playwright respectively) One next step means one operation within the three.'''

action_format = "ACTION: Choose an action from {CLICK, TYPE, SELECT}."

value_format = "VALUE: Provide additional input based on ACTION.\n\nThe VALUE means:\nIf ACTION == TYPE, specify the " \
               "text to be typed.\nIf ACTION == SELECT, specify the option to be chosen.\nIf ACTION == CLICK, " \
               "write \"None\"."

question_description_new_exp4 = '''The screenshot below shows the webpage you see. Follow the following guidance to think step by step before outlining the next action step at the current stage:

(Current Webpage Identification)
Firstly, think about what the current webpage is.

(Previous Action Analysis)
Secondly, combined with the screenshot, analyze each step of the previous action history and their intention one by one. Particularly, pay more attention to the last step, which may be more related to what you should do now as the next step.

(Screenshot Details Analysis)
Closely examine the screenshot to check the status of every part of the webpage to understand what you can operate with and what has been set or completed. You should closely examine the screenshot details to see what steps have been completed by previous actions even though you are given the textual previous actions. Because the textual history may not clearly and sufficiently record some effects of previous actions, you should closely evaluate the status of every part of the webpage to understand what you have done.

(Next Action Based on Webpage and Analysis)
Then, based on your analysis, in conjunction with human web browsing habits and the logic of web design, decide on the following action. And clearly outline which element in the webpage users will operate with as the first next target element, its detailed location, and the corresponding operation.

To be successful, it is important to follow the following rules: 
1. You should only issue a valid action given the current observation. 
2. You should only issue one action at a time'''

question_description_new_exp2 = '''The screenshot below shows the webpage you see. In the screenshot, some red bounding boxes and white-on-black uppercase letters at the bottom left corner of the bounding boxes have been manually added. You should ignore them for now. Follow the following guidance to think step by step before outlining the next action step at the current stage:

(Current Webpage Identification)
Firstly, think about what the current webpage is.

(Previous Action Analysis)
Secondly, combined with the screenshot, analyze each step of the previous action history and their intention one by one. Particularly, pay more attention to the last step, which may be more related to what you should do now as the next step.

(Screenshot Details Analysis)
Closely examine the screenshot to check the status of every part of the webpage to understand what you can operate with and what has been set or completed. You should closely examine the screenshot details to see what steps have been completed by previous actions even though you are given the textual previous actions. Because the textual history may not clearly and sufficiently record some effects of previous actions, you should closely evaluate the status of every part of the webpage to understand what you have done.

(Next Action Based on Webpage and Analysis)
Then, based on your analysis, in conjunction with human web browsing habits and the logic of web design, decide on the following action. And clearly outline which element in the webpage users will operate with as the first next target element, its detailed location, and the corresponding operation.

To be successful, it is important to follow the following rules: 
1. You should only issue a valid action given the current observation. 
2. You should only issue one action at a time.'''

question_description_new_exp3 = '''The screenshot below shows the webpage you see. Follow the following guidance to think step by step before outlining the next action step at the current stage:

(Current Webpage Identification)
Firstly, think about what the current webpage is.

(Previous Action Analysis)
Secondly, combined with the screenshot, analyze each step of the previous action history and their intention one by one. Particularly, pay more attention to the last step, which may be more related to what you should do now as the next step.

(Screenshot Details Analysis)
Closely examine the screenshot to check the status of every part of the webpage to understand what you can operate with and what has been set or completed. You should closely examine the screenshot details to see what steps have been completed by previous actions even though you are given the textual previous actions. Because the textual history may not clearly and sufficiently record some effects of previous actions, you should closely evaluate the status of every part of the webpage to understand what you have done.

(Next Action Based on Webpage and Analysis)
Then, based on your analysis, in conjunction with human web browsing habits and the logic of web design, decide on the following action. And clearly outline which element in the webpage users will operate with as the first next target element, its detailed location, and the corresponding operation. Please also closely examine the screenshot to adequately describe its position relative to nearby elements and its textual or visual content (if it has). If you find multiple elements similar to your target element, use a more precise description to ensure people can distinguish your target element from them through your answer.

To be successful, it is important to follow the following rules: 
1. You should only issue a valid action given the current observation. 
2. You should only issue one action at a time.'''

exp4_prompt_dict = {
    "system_prompt": sys_prompt,

    "question_description": question_description_new_exp4,

    "referring_description": f"""(Reiteration)
First, reiterate your next target element, its detailed location, and the corresponding operation.

(Multichoice Question)
Below is a multi-choice question, where the choices are elements in the webpage. From the screenshot, find out where and what each one is on the webpage. Then, determine whether one matches your target element. Please examine the choices one by one. Choose the matching one. If multiple options match your answer, choose the most likely one by re-examining the screenshot, the choices, and your further reasoning.""",

    "element_format": """(Final Answer)
Finally, conclude your answer using the format below. Ensure your answer is strictly adhering to the format provided below. Please do not leave any explanation in your answers of the final standardized format part, and this final part should be clear and certain. The element choice, action, and value should be in three separate lines.

Format:

ELEMENT: The uppercase letter of your choice.""",

    "action_format": f"{action_format}",

    "value_format": f"{value_format}"
}

exp2_prompt_dict = {
    "system_prompt": sys_prompt,

    "question_description": question_description_new_exp2,

    "referring_description": f"""(Reiteration)
First, reiterate your next target element, its detailed location, and the corresponding operation.

(Verification with the Screenshot)
Then, please closely re-examine the screenshot to find whether your target element is marked by a red bounding box and has a white uppercase letter on a black background at the bottom left corner of the bounding box, which is positioned closely next to the bounding box. If yes, use that letter for your final answer. If not, please do not make them up. If it is not marked, please output "NA" as your target element in the following final answer part.""",

    "element_format": """(Final Answer)
Finally, conclude your answer using the format below. Ensure your answer is strictly adhering to the format provided below. Please do not leave any explanation in your answers of the final standardized format part, and this final part should be clear and certain. The element choice, action, and value should be in three separate lines.

Format:

ELEMENT: The uppercase letter of your choice.""",

    "action_format": f"{action_format}",

    "value_format": f"{value_format}"
}

exp3_prompt_dict = {
    "system_prompt": sys_prompt,

    "question_description": question_description_new_exp3,

    "referring_description": f"""""",

    "element_format": """(Final Answer)
Finally, conclude your answer using the format below. Ensure your answer is strictly adhering to the format provided below. Please do not leave any explanation in your answers of the final standardized format part, and this final part should be clear and certain. The element, element type, element text, action and value should be in five separate lines.

Format:

ELEMENT: Please describe which element you need to operate with. Describe it as detailed as possible, including what it is and where it is.

ELEMENT TYPE: Please specify its type from these options: BUTTON, TEXTBOX, SELECTBOX, or LINK.

ELEMENT TEXT: Please provide the exact text displayed on the element. Do not invent or modify the text; reproduce it as-is from the screenshot.""",

    "action_format": f"{action_format}",

    "value_format": f"{value_format}"
}




##### SeeAct Online Prompts

seeact_online_sys_prompt = '''Imagine that you are imitating humans doing web navigation for a task step by step. At each stage, you can see the webpage like humans by a screenshot and know the previous actions before the current step decided by yourself through recorded history. You need to decide on the first following action to take. You can click on an element with the mouse, select an option, type text or press Enter with the keyboard. (For your understanding, they are like the click(), select_option() type() and keyboard.press('Enter') functions in playwright respectively) One next step means one operation within the four. Unlike humans, for typing (e.g., in text areas, text boxes) and selecting (e.g., from dropdown menus or <select> elements), you should try directly typing the input or selecting the choice, bypassing the need for an initial click. You should not attempt to create accounts, log in or do the final submission. Terminate when you deem the task complete or if it requires potentially harmful actions.'''

seeact_online_question_description_new_exp4 = '''The screenshot below shows the webpage you see. Follow the following guidance to think step by step before outlining the next action step at the current stage:

(Current Webpage Identification)
Firstly, think about what the current webpage is.

(Previous Action Analysis)
Secondly, combined with the screenshot, analyze each step of the previous action history and their intention one by one. Particularly, pay more attention to the last step, which may be more related to what you should do now as the next step. Specifically, if the last action involved a TYPE, always evaluate whether it necessitates a confirmation step, because typically a single TYPE action does not make effect. (often, simply pressing 'Enter', assuming the default element involved in the last action, unless other clear elements are present for operation).

(Screenshot Details Analysis)
Closely examine the screenshot to check the status of every part of the webpage to understand what you can operate with and what has been set or completed. You should closely examine the screenshot details to see what steps have been completed by previous actions even though you are given the textual previous actions. Because the textual history may not clearly and sufficiently record some effects of previous actions, you should closely evaluate the status of every part of the webpage to understand what you have done.

(Next Action Based on Webpage and Analysis)
Then, based on your analysis, in conjunction with human web browsing habits and the logic of web design, decide on the following action. And clearly outline which element in the webpage users will operate with as the first next target element, its detailed location, and the corresponding operation.

To be successful, it is important to follow the following rules: 
1. You should only issue a valid action given the current observation. 
2. You should only issue one action at a time
3. For handling the select dropdown elements on the webpage, it's not necessary for you to provide completely accurate options right now. The full list of options for these elements will be supplied later.'''

seeact_online_action_format = "ACTION: Choose an action from {CLICK, SELECT, TYPE, PRESS ENTER, TERMINATE, NONE}."

seeact_online_value_format = "VALUE: Provide additional input based on ACTION.\n\nThe VALUE means:\nIf ACTION == TYPE, specify the " \
               "text to be typed.\nIf ACTION == SELECT, indicate the option to be chosen. Revise the selection value to align with the available options within the element.\nIf ACTION == CLICK, PRESS ENTER, TERMINATE or NONE, " \
               "write \"None\"."

seeact_choice_prompt_dict = {
    "system_prompt": seeact_online_sys_prompt,

    "question_description": seeact_online_question_description_new_exp4,

    "referring_description": f"""(Reiteration)
First, reiterate your next target element, its detailed location, and the corresponding operation.

(Multichoice Question)
Below is a multi-choice question, where the choices are elements in the webpage. All elements are arranged in the order based on their height on the webpage, from top to bottom (and from left to right). This arrangement can be used to locate them. From the screenshot, find out where and what each one is on the webpage, taking into account both their text content and HTML details. Then, determine whether one matches your target element. Please examine the choices one by one. Choose the matching one. If multiple options match your answer, choose the most likely one by re-examining the screenshot, the choices, and your further reasoning.""",

    "element_format": """(Final Answer)
Finally, conclude your answer using the format below. Ensure your answer is strictly adhering to the format provided below. Please do not leave any explanation in your answers of the final standardized format part, and this final part should be clear and certain. The element choice, action, and value should be in three separate lines.

Format:

ELEMENT: The uppercase letter of your choice. (No need for PRESS ENTER)""",

    "action_format": f"{seeact_online_action_format}",

    "value_format": f"{seeact_online_value_format}"
}

















def generate_prompt(experiment_split, task=None, previous=None, choices=None):
    assert experiment_split != None, "Please specify the experiment split."
    assert task != None, "Please input the task."
    assert previous != None, "Please input the previous actions."

    prompt_list = []
    system_prompt_input = None
    question_description_input = None
    referring_input = None
    element_format_input = None
    action_format_input = None
    value_format_input = None

    if experiment_split in ["text","text_choice","4api", "4api_aug"]:
        system_prompt_input = exp4_prompt_dict["system_prompt"]
        question_description_input = exp4_prompt_dict["question_description"]
        referring_input = exp4_prompt_dict["referring_description"]
        element_format_input = exp4_prompt_dict["element_format"]
        action_format_input = exp4_prompt_dict["action_format"]
        value_format_input = exp4_prompt_dict["value_format"]

        prompt_list.extend(
            generate_new_query_prompt(system_prompt=system_prompt_input, task=task, previous_actions=previous,
                                      question_description=question_description_input))
        prompt_list.append(
            generate_new_referring_prompt(referring_description=referring_input, element_format=element_format_input,
                                          action_format=action_format_input, value_format=value_format_input,
                                          choices=choices))
        
        prompt_list[1] = prompt_list[1] + "\n3. If the target element is a text input field such as a search bar, directly issue ACTION=TYPE with the desired text, instead of ACTION=CLICK."
        
        return prompt_list

    elif experiment_split in ["element_attributes","3api"]:
        system_prompt_input = exp3_prompt_dict["system_prompt"]
        question_description_input = exp3_prompt_dict["question_description"]
        referring_input = exp3_prompt_dict["referring_description"]
        element_format_input = exp3_prompt_dict["element_format"]
        action_format_input = exp3_prompt_dict["action_format"]
        value_format_input = exp3_prompt_dict["value_format"]

        prompt_list.extend(
            generate_new_query_prompt(system_prompt=system_prompt_input, task=task, previous_actions=previous,
                                      question_description=question_description_input))
        prompt_list.append(
            generate_new_referring_prompt(referring_description=referring_input, element_format=element_format_input,
                                          action_format=action_format_input, value_format=value_format_input,
                                          split="3api"
                                          ))
        prompt_list[1] = prompt_list[1] + "\n3. If the target element is a text input field such as a search bar, directly issue ACTION=TYPE with the desired text, instead of ACTION=CLICK."
        
        return prompt_list

    elif experiment_split in ["image_annotation","2api"]:
        system_prompt_input = exp2_prompt_dict["system_prompt"]
        question_description_input = exp2_prompt_dict["question_description"]
        referring_input = exp2_prompt_dict["referring_description"]
        element_format_input = exp2_prompt_dict["element_format"]
        action_format_input = exp2_prompt_dict["action_format"]
        value_format_input = exp2_prompt_dict["value_format"]

        prompt_list.extend(
            generate_new_query_prompt(system_prompt=system_prompt_input, task=task, previous_actions=previous,
                                      question_description=question_description_input))
        prompt_list.append(
            generate_new_referring_prompt(referring_description=referring_input, element_format=element_format_input,
                                          action_format=action_format_input, value_format=value_format_input,
                                          choices=None))
        prompt_list[1] = prompt_list[1] + "\n3. If the target element is a text input field such as a search bar, directly issue ACTION=TYPE with the desired text, instead of ACTION=CLICK."
        
        return prompt_list
    elif experiment_split in ["seeact_online","online","seeact","SeeAct"]:
        system_prompt_input = seeact_choice_prompt_dict["system_prompt"]
        question_description_input = seeact_choice_prompt_dict["question_description"]
        referring_input = seeact_choice_prompt_dict["referring_description"]
        element_format_input = seeact_choice_prompt_dict["element_format"]
        action_format_input = seeact_choice_prompt_dict["action_format"]
        value_format_input = seeact_choice_prompt_dict["value_format"]
        prompt_list = []

        prompt_list.extend(
            generate_new_query_prompt(system_prompt=system_prompt_input, task=task, previous_actions=previous,
                                      question_description=question_description_input))
        prompt_list.append(
            generate_new_referring_prompt(referring_description=referring_input, element_format=element_format_input,
                                          action_format=action_format_input, value_format=value_format_input,
                                          choices=choices))
        prompt_list[1] = prompt_list[1] + "\n3. If the target element is a text input field such as a search bar, directly issue ACTION=TYPE with the desired text, instead of ACTION=CLICK."
        
        return prompt_list

