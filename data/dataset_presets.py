def get_presets():
    presets = {}

    presets['formal_fallacies_syllogisms_negation'] = dict(
        dataset_name = "formal_fallacies_syllogisms_negation",
        dataset_config = "main",
        format = "bigbench",
        prompt = "[INST] {inputs} [/INST]\n",
        response = "{targets}",
    )

    presets['language_identification'] = dict(
        dataset_name = "language_identification",
        dataset_config = "main",
        format = "bigbench",
        prompt = "[INST] {inputs} [/INST]\n",
        response = "{targets}",
    )

    presets['linguistics_puzzles'] = dict(
        dataset_name = "linguistics_puzzles",
        dataset_config = "main",
        format = "bigbench",
        prompt = "[INST] {inputs} [/INST]\n",
        response = "{targets}",
    )

    presets['logical_deduction'] = dict(
        dataset_name = "logical_deduction",
        dataset_config = "main",
        format = "bigbench",
        prompt = "[INST] {inputs} [/INST]\n",
        response = "{targets}",
    )

    presets['play_dialog_same_or_different'] = dict(
        dataset_name = "play_dialog_same_or_different",
        dataset_config = "main",
        format = "bigbench",
        prompt = "[INST] {inputs} [/INST]\n",
        response = "{targets}",
    )

    presets['strategyqa'] = dict(
        dataset_name = "strategyqa",
        dataset_config = "main",
        format = "bigbench",
        prompt = "[INST] {inputs} [/INST]\n",
        response = "{targets}",
    )

    presets['vitaminc_fact_verification'] = dict(
        dataset_name = "vitaminc_fact_verification",
        dataset_config = "main",
        format = "bigbench",
        prompt = "[INST] {inputs} [/INST]\n",
        response = "{targets}",
    )

    presets['winowhy'] = dict(
        dataset_name = "winowhy",
        dataset_config = "main",
        format = "bigbench",
        prompt = "[INST] {inputs} [/INST]\n",
        response = "{targets}",
    )

    presets['gsm8k'] = dict(
        dataset_name = "gsm8k",
        subset_name = "main",
        dataset_config = "main",
        format = "hf",
        prompt = "[INST] {question} [/INST]\n",
        response = "{answer}",
    )

    presets['object_counting'] = dict(
        dataset_name = "object_counting",
        dataset_config = "main",
        format = "bigbench",
        prompt = "[INST] {inputs} [/INST]\n",
        response = "{targets}",
    )

    presets['abstract_narrative_understanding'] = dict(
        dataset_name = "abstract_narrative_understanding",
        dataset_config = "main",
        format = "bigbench",
        prompt = "[INST] {inputs} [/INST]\n",
        response = "{targets}",
    )

    presets['contextual_parametric_knowledge_conflicts'] = dict(
        dataset_name = "contextual_parametric_knowledge_conflicts",
        dataset_config = "main",
        format = "bigbench",
        prompt = "[INST] {inputs} [/INST]\n",
        response = "{targets}",
    )

    presets['cs_algorithms'] = dict(
        dataset_name = "cs_algorithms",
        dataset_config = "main",
        format = "bigbench",
        prompt = "[INST] {inputs} [/INST]\n",
        response = "{targets}",
    )

    presets['elementary_math_qa'] = dict(
        dataset_name = "elementary_math_qa",
        dataset_config = "main",
        format = "bigbench",
        prompt = "[INST] {inputs} [/INST]\n",
        response = "{targets}",
    )

    presets['question_selection'] = dict(
        dataset_name = "question_selection",
        dataset_config = "main",
        format = "bigbench",
        prompt = "[INST] {inputs} [/INST]\n",
        response = "{targets}",
    )

    presets['topical_chat'] = dict(
        dataset_name = "topical_chat",
        dataset_config = "main",
        format = "bigbench",
        prompt = "[INST] {inputs} [/INST]\n",
        response = "{targets}",
    )

    presets['cnn_dailymail'] = dict(
        dataset_name = "cnn_dailymail",
        subset_name = '3.0.0',
        dataset_config = "main",
        format = "hf",
        prompt = "[INST] Generate a summarization of the news article.\n{article} [/INST]\n",
        response = "{highlights}",
    )

    presets['news_commentary_es'] = dict(
        dataset_name = "Helsinki-NLP/news_commentary",
        subset_name = 'en-es',
        dataset_config = "main",
        format = "hf",
        prompt = "[INST] Traducir las siguientes frases al inglés.\n{es} [/INST]\n",
        response = "{en}",
    )

    presets['news_commentary_it'] = dict(
        dataset_name = "Helsinki-NLP/news_commentary",
        subset_name = 'en-it',
        dataset_config = "main",
        format = "hf",
        prompt = "[INST] Tradurre le seguenti frasi in inglese.\n{it} [/INST]\n",
        response = "{en}",
    )

    presets['news_commentary_de'] = dict(
        dataset_name = "Helsinki-NLP/news_commentary",
        subset_name = 'de-en',
        dataset_config = "main",
        format = "hf",
        prompt = "[INST] Den folgenden Text ins Englische übersetzen.\n{de} [/INST]\n",
        response = "{en}",
    )

    presets['tracking_shuffled_objects'] = dict(
        dataset_name = "tracking_shuffled_objects",
        dataset_config = "main",
        format = "bigbench",
        prompt = "[INST] {inputs} [/INST]\n",
        response = "{targets}",
    )

    presets['goal_step_wikihow'] = dict(
        dataset_name = "goal_step_wikihow",
        dataset_config = "main",
        format = "bigbench",
        prompt = "[INST] {inputs} [/INST]\n",
        response = "{targets}",
    )

    presets['disfl_qa'] = dict(
        dataset_name = "disfl_qa",
        dataset_config = "main",
        format = "bigbench",
        prompt = "[INST] {inputs} [/INST]\n",
        response = "{targets}",
    )

    presets['unit_conversion'] = dict(
        dataset_name = "unit_conversion",
        dataset_config = "main",
        format = "bigbench",
        prompt = "[INST] {inputs} [/INST]\n",
        response = "{targets}",
    )

    presets['paragraph_segmentation'] = dict(
        dataset_name = "paragraph_segmentation",
        dataset_config = "main",
        format = "bigbench",
        prompt = "[INST] {inputs} [/INST]\n",
        response = "{targets}",
    )

    presets['reasoning_about_colored_objects'] = dict(
        dataset_name = "reasoning_about_colored_objects",
        dataset_config = "main",
        format = "bigbench",
        prompt = "[INST] {inputs} [/INST]\n",
        response = "{targets}",
    )

    presets['epistemic_reasoning'] = dict(
        dataset_name = "epistemic_reasoning",
        dataset_config = "main",
        format = "bigbench",
        prompt = "[INST] {inputs} [/INST]\n",
        response = "{targets}",
    )

    presets['alpaca'] = dict(
        dataset_name = "tatsu-lab/alpaca",
        subset_name = "default",
        dataset_config = "main",
        format = "hf",
        prompt = "[INST] {instruction} [/INST]\n",
        response = "{output}",
    )

    return presets

news_commentary = {'news_commentary_es', 'news_commentary_it', 'news_commentary_de'}