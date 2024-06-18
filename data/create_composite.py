import random
import os
import argparse
import json

random.seed(47)

###
dataset_dir = 'datasets'
target_base_dir = 'datasets'
###

composite_3_tasks = ['logical_deduction', 'question_selection', 'strategyqa']
composite_5_tasks = ['logical_deduction', 'question_selection', 'abstract_narrative_understanding', 'goal_step_wikihow', 'strategyqa']
composite_10_tasks = ["logical_deduction", "question_selection", "abstract_narrative_understanding", "goal_step_wikihow", "winowhy", "strategyqa", "disfl_qa", "news_commentary_de", "alpaca", "linguistics_puzzles"]

composite_3_prefix = """
Example 1:

[INST] The following paragraphs each describe a set of five objects arranged in a fixed order. The statements are logically consistent within each paragraph.

On a branch, there are five birds: a cardinal, a crow, a falcon, a robin, and a blue jay. The robin is to the right of the cardinal. The cardinal is to the right of the blue jay. The blue jay is the second from the left. The crow is the rightmost. Which choice is correct?\nchoice: The cardinal is the leftmost.
choice: The crow is the leftmost.
choice: The falcon is the leftmost.
choice: The robin is the leftmost.
choice: The blue jay is the leftmost.
[/INST]

[INST] Given a short answer along with its context, select the most appropriate question which has the given short answer as its answer.

Here is the short answer followed by the context.
Short Answer: magazines and journals Context: Tesla wrote a number of books and articles for magazines and journals. Among his books are My Inventions: The Autobiography of Nikola Tesla, compiled and edited by Ben Johnston; The Fantastic Inventions of Nikola Tesla, compiled and edited by David Hatcher Childress; and The Tesla Papers.
choice: Who was the first to post tesla's writings?
choice: Who was in charge of editing tesla's autobiography?
Choose the appropriate question which has the given answer. [/INST]

[INST] Q: Can Hulk's alter ego explain atomic events?
A: [/INST]

1. The falcon is the leftmost.
2. Who was the first to post tesla's writings?
3. Yes. Hulk's alter ego is Dr. Robert Bruce Banner. Dr. Robert Bruce Banner is a nuclear physicist.  Nuclear physics is the field of physics that studies atomic nuclei and their constituents and interactions.


Example 2:

[INST] The following paragraphs each describe a set of three objects arranged in a fixed order. The statements are logically consistent within each paragraph.

In a golf tournament, there were three golfers: Ana, Rob, and Joe. Joe finished above Ana. Rob finished above Joe. Which choice is correct?
choice: Ana finished second.
choice: Rob finished second.
choice: Joe finished second.
[/INST]

[INST] Given a short answer along with its context, select the most appropriate question which has the given short answer as its answer.

Here is the short answer followed by the context.
Short Answer: two months Context: It was not until January 1518 that friends of Luther translated the 95 Theses from Latin into German and printed and widely copied them, making the controversy one of the first in history to be aided by the printing press. Within two weeks, copies of the theses had spread throughout Germany; within two months, they had spread throughout Europe.
choice: How long did the theses take to spread through europe?
choice: How long did it take for the printing of the theses to spread through germany?
Choose the appropriate question which has the given answer. [/INST]

[INST] Q: Can Iowa be hidden in the English Channel?
A: [/INST]

1. Joe finished second.
2. How long did the theses take to spread through europe?
3. No. The maximum width of the English Channel is 150 miles. The minimum width of Iowa is 200 miles.


Following the example above, generate answers to the questions below:

"""

composite_5_prefix = """
Example 1:

[INST] The following paragraphs each describe a set of five objects arranged in a fixed order. The statements are logically consistent within each paragraph.

On a branch, there are five birds: a cardinal, a crow, a falcon, a robin, and a blue jay. The robin is to the right of the cardinal. The cardinal is to the right of the blue jay. The blue jay is the second from the left. The crow is the rightmost. Which choice is correct?\nchoice: The cardinal is the leftmost.
choice: The crow is the leftmost.
choice: The falcon is the leftmost.
choice: The robin is the leftmost.
choice: The blue jay is the leftmost.
[/INST]

[INST] Given a short answer along with its context, select the most appropriate question which has the given short answer as its answer.

Here is the short answer followed by the context.
Short Answer: magazines and journals Context: Tesla wrote a number of books and articles for magazines and journals. Among his books are My Inventions: The Autobiography of Nikola Tesla, compiled and edited by Ben Johnston; The Fantastic Inventions of Nikola Tesla, compiled and edited by David Hatcher Childress; and The Tesla Papers.
choice: Who was the first to post tesla's writings?
choice: Who was in charge of editing tesla's autobiography?
Choose the appropriate question which has the given answer. [/INST]

[INST] In what follows, we provide short narratives, each of which illustrates a common proverb. 
Narrative: One of my students invited me on the opening of his new coffee shop business. I advised him that never comprise on quality of products and services as once people don't like anything they will never come back.
This narrative is a good illustration of the following proverb:
choice: Honey catches more flies than vinegar
choice: A barking dog never bites
choice: Flattery will get you nowhere
choice: Don't shoot the messenger
choice: Once bitten, twice shy
[/INST]

[INST] Q: In order to 'Import China Teas Into USA', one can 'Form a Company' and 'Register with FDA as an importer'. Which should happen first? 
choice: Register with FDA as an importer.
choice: Form a Company.
A: [/INST]

[INST] Q: Can Hulk's alter ego explain atomic events?
A: [/INST]

1. The falcon is the leftmost.
2. Who was the first to post tesla's writings?
3. Once bitten, twice shy
4. Form a Company.
5. Yes. Hulk's alter ego is Dr. Robert Bruce Banner. Dr. Robert Bruce Banner is a nuclear physicist.  Nuclear physics is the field of physics that studies atomic nuclei and their constituents and interactions.


Example 2:

[INST] The following paragraphs each describe a set of three objects arranged in a fixed order. The statements are logically consistent within each paragraph.

In a golf tournament, there were three golfers: Ana, Rob, and Joe. Joe finished above Ana. Rob finished above Joe. Which choice is correct?
choice: Ana finished second.
choice: Rob finished second.
choice: Joe finished second.
[/INST]

[INST] Given a short answer along with its context, select the most appropriate question which has the given short answer as its answer.

Here is the short answer followed by the context.
Short Answer: two months Context: It was not until January 1518 that friends of Luther translated the 95 Theses from Latin into German and printed and widely copied them, making the controversy one of the first in history to be aided by the printing press. Within two weeks, copies of the theses had spread throughout Germany; within two months, they had spread throughout Europe.
choice: How long did the theses take to spread through europe?
choice: How long did it take for the printing of the theses to spread through germany?
Choose the appropriate question which has the given answer. [/INST]

[INST] In what follows, we provide short narratives, each of which illustrates a common proverb. 
Narrative: David wants to be well liked by his peers, so acts and behaves in such a way that he will hopefully be treated equally.
This narrative is a good illustration of the following proverb:
choice: Never judge a book by its cover
choice: A barking dog never bites
choice: A prophet is not recognized in his own land
choice: Do unto others as you would have them do to you
choice: Give credit where credit is due
[/INST]

[INST] Q: The most reasonable step of 'Photograph Star Trails' is 
choice: Fill with potting soil
choice: Understand the premise of Fortnite
choice: Dig a hole about the same size as the grass plug's root ball
choice: Place the camera on a sturdy base
A: [/INST]

[INST] Q: Can Iowa be hidden in the English Channel?
A: [/INST]

1. Joe finished second.
2. How long did the theses take to spread through europe?
3. Do unto others as you would have them do to you
4. Place the camera on a sturdy base.
5. No. The maximum width of the English Channel is 150 miles. The minimum width of Iowa is 200 miles.


Following the example above, generate answers to the questions below:

"""

composite_10_prefix = """
Example 1:

[INST] The following paragraphs each describe a set of five objects arranged in a fixed order. The statements are logically consistent within each paragraph.

On a branch, there are five birds: a cardinal, a crow, a falcon, a robin, and a blue jay. The robin is to the right of the cardinal. The cardinal is to the right of the blue jay. The blue jay is the second from the left. The crow is the rightmost. Which choice is correct?\nchoice: The cardinal is the leftmost.
choice: The crow is the leftmost.
choice: The falcon is the leftmost.
choice: The robin is the leftmost.
choice: The blue jay is the leftmost.
[/INST]

[INST] Given a short answer along with its context, select the most appropriate question which has the given short answer as its answer.

Here is the short answer followed by the context.
Short Answer: magazines and journals Context: Tesla wrote a number of books and articles for magazines and journals. Among his books are My Inventions: The Autobiography of Nikola Tesla, compiled and edited by Ben Johnston; The Fantastic Inventions of Nikola Tesla, compiled and edited by David Hatcher Childress; and The Tesla Papers.
choice: Who was the first to post tesla's writings?
choice: Who was in charge of editing tesla's autobiography?
Choose the appropriate question which has the given answer. [/INST]

[INST] In what follows, we provide short narratives, each of which illustrates a common proverb. 
Narrative: One of my students invited me on the opening of his new coffee shop business. I advised him that never comprise on quality of products and services as once people don't like anything they will never come back.
This narrative is a good illustration of the following proverb:
choice: Honey catches more flies than vinegar
choice: A barking dog never bites
choice: Flattery will get you nowhere
choice: Don't shoot the messenger
choice: Once bitten, twice shy
[/INST]

[INST] Q: In order to 'Import China Teas Into USA', one can 'Form a Company' and 'Register with FDA as an importer'. Which should happen first? 
choice: Register with FDA as an importer.
choice: Form a Company.
A: [/INST]

[INST] Please answer the following questions about which words certain pronouns refer to.

I can't cut that tree down with that axe; it is too small. The 'it' refers to the axe because The axe is smaller to cut the tree.
The above reasoning is [/INST]

[INST] Q: Can Hulk's alter ego explain atomic events?
A: [/INST]

[INST] Question: In the most basic sense what did a researchers wait no Turing machine emulate ?  
context: Before the actual research explicitly devoted to the complexity of algorithmic problems started off , numerous foundations were laid out by various researchers . Most influential among these was the definition of Turing machines by Alan Turing in 1936 , which turned out to be a very robust and flexible simplification of a computer . 
Answer: [/INST]

[INST] Den folgenden Text ins Englische \u00fcbersetzen.\nNat\u00fcrlich kann das auch in Demokratien passieren. [/INST]

[INST] Suggest a title for an article about ransomware attacks.
[/INST]

[INST] English: small people that see short crows slapped slow birds Other: woci wuli koxepu gepameso du woci xapi xifupu woci soqo sowapu yicafaso
English: the slow birds will see a small crow Other: ziye soqo sowapu woci xapi koxevu gepabeso
English: the slow bird will remember fabulous cats Other: ziye soqo sowavu woci neza toyepu nafebe
English: the big person remembers a slow bird that fabulous cats dislike Other: ziye luhu xifuvu woci neza toyepu xatumeso du woci soqo sowavu nafeme
English: short monkeys dislike the slow cats Other: woci wuli ramipu ziye soqo toyepu xatumeso
English: a short crow will dislike a small monkey Other: [/INST]

1. The falcon is the leftmost.
2. Who was the first to post tesla's writings?
3. Once bitten, twice shy
4. Form a Company.
5. correct
6. Yes. Hulk's alter ego is Dr. Robert Bruce Banner. Dr. Robert Bruce Banner is a nuclear physicist.  Nuclear physics is the field of physics that studies atomic nuclei and their constituents and interactions.
7. a computer
8. Of course, democracies can be captured too.
9. \"The Growing Threat of Ransomware Attacks: What You Need to Know\"
10. woci wuli koxevu woci xapi ramivu xatube


Example 2:

[INST] The following paragraphs each describe a set of three objects arranged in a fixed order. The statements are logically consistent within each paragraph.

In a golf tournament, there were three golfers: Ana, Rob, and Joe. Joe finished above Ana. Rob finished above Joe. Which choice is correct?
choice: Ana finished second.
choice: Rob finished second.
choice: Joe finished second.
[/INST]

[INST] Given a short answer along with its context, select the most appropriate question which has the given short answer as its answer.

Here is the short answer followed by the context.
Short Answer: two months Context: It was not until January 1518 that friends of Luther translated the 95 Theses from Latin into German and printed and widely copied them, making the controversy one of the first in history to be aided by the printing press. Within two weeks, copies of the theses had spread throughout Germany; within two months, they had spread throughout Europe.
choice: How long did the theses take to spread through europe?
choice: How long did it take for the printing of the theses to spread through germany?
Choose the appropriate question which has the given answer. [/INST]

[INST] In what follows, we provide short narratives, each of which illustrates a common proverb. 
Narrative: David wants to be well liked by his peers, so acts and behaves in such a way that he will hopefully be treated equally.
This narrative is a good illustration of the following proverb:
choice: Never judge a book by its cover
choice: A barking dog never bites
choice: A prophet is not recognized in his own land
choice: Do unto others as you would have them do to you
choice: Give credit where credit is due
[/INST]

[INST] Q: The most reasonable step of 'Photograph Star Trails' is 
choice: Fill with potting soil
choice: Understand the premise of Fortnite
choice: Dig a hole about the same size as the grass plug's root ball
choice: Place the camera on a sturdy base
A: [/INST]

[INST] Please answer the following questions about which words certain pronouns refer to.

Ann asked Mary what time the library closes, because she had forgotten. The 'she' refers to ann because Ann already didn't know.
The above reasoning is [/INST]

[INST] Q: Can Iowa be hidden in the English Channel?
A: [/INST]

[INST] Question: When was the Latin version of the word Norman first heard no I mean recorded ?  
context: The English name \" Normans \" comes from the French words Normans / Normanz , plural of Normant , modern French normand , which is itself borrowed from Old Low Franconian Nortmann \" Northman \" or directly from Old Norse Nor\u00f0ma\u00f0r , Latinized variously as Nortmannus , Normannus , or Nordmannus ( recorded in Medieval Latin , 9th century ) to mean \" Norseman , Viking \" . 
Answer: [/INST]

[INST] Den folgenden Text ins Englische \u00fcbersetzen.\nBei stabilen und vorhersehbaren Wechselkursen ist der Carry Trade ein relativ sicheres Gesch\u00e4ft. [/INST]

[INST] Imagine you are a programmer assigned to do this task. Analyze the instruction and design a program to meet the requirements.
Design a program that can take a list of numbers and sort them in ascending order. [/INST]

[INST] Other: xexu taxiti tatiho xacici datuji xijagu xilagemo English: the tall dog touches slow crows
Other: xexu xixiti xixuqo baxazeko datuji tapiya talagemo English: the tall people dislike a big crow
Other: xexu xigopa xiduni baxazeko datuji xijagu xiyaqamo English: the fast birds dislike slow cats
Other: xexu tajagu taduni xazeko xexuji xigopa xiyaqamo English: the slow bird dislikes the fast cats
Other: datu xifani xilage batopuse datuji tajagu taxuqomo English: excellent crows will praise a slow person
Other: datu xijagu xixuqo baxazeko datuji xifani xidunimo English: [/INST]

1. Joe finished second.
2. How long did the theses take to spread through europe?
3. Do unto others as you would have them do to you
4. Place the camera on a sturdy base.
5. incorrect
6. No. The maximum width of the English Channel is 150 miles. The minimum width of Iowa is 200 miles.
7. 9th century
8. When exchange rates are stable and predictable, the carry trade is relatively safe.
9. The program should take a list of numbers as input, sort the numbers in ascending order, and output the sorted list. The program should include a loop to traverse the list and a comparison function to compare two elements and determine their order.
10. slow people dislike excellent birds


Following the example above, generate answers to the questions below:

"""

def composite_task(args):
    if args.n == 3:
        tasks = composite_3_tasks
        prefix = composite_3_prefix
    elif args.n == 5:
        tasks = composite_5_tasks
        prefix = composite_5_prefix
    elif args.n == 10:
        tasks = composite_10_tasks
        prefix = composite_10_prefix
    else:
        raise ValueError('Invalid number of tasks')

    source_task_paths = [os.path.join(dataset_dir, task, 'test.jsonl') for task in tasks]
    target_file = os.path.join(target_base_dir, f"composite_{len(tasks)}", 'test.jsonl')
    os.makedirs(os.path.dirname(target_file), exist_ok=True)
    source_files = [open(source_task, 'r', encoding='utf-8') for source_task in source_task_paths]

    with open(target_file, 'w', encoding='utf-8') as target_file:
        # tasks_samples: (n, sample_num)
        tasks_samples = [source_file.readlines() for source_file in source_files]
        for task_samples in tasks_samples:
            random.shuffle(task_samples)
            task_samples = task_samples[:args.nums]

        for i in range(args.nums):
            tasks_sample = [json.loads(task_samples[i]) for task_samples in tasks_samples]
            composite_prompt = prefix
            composite_response = ''
            for i, task_sample in enumerate(tasks_sample):
                composite_prompt += task_sample['prompt'] + '\n\n'
                composite_response += task_sample['response'] + '@@@@\n'
            # delete the last '\n'
            composite_prompt = composite_prompt[:-1]
            # delete the last '@@@@\n'
            composite_response = composite_response[:-5]
            target_file.write(json.dumps({'prompt': composite_prompt, 'response': composite_response}) + '\n')

    for source_file in source_files:
        source_file.close()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--n', type=int, required=True)
    argparser.add_argument('--nums', default=200, type=int, required=False)
    args = argparser.parse_args()
    composite_task(args)