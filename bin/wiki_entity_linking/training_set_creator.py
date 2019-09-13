# coding: utf-8
from __future__ import unicode_literals

import random
import re
import bz2
import datetime

from spacy.gold import GoldParse
from bin.wiki_entity_linking import kb_creator

"""
Process Wikipedia interlinks to generate a training dataset for the EL algorithm.
Gold-standard entities are stored in one file in standoff format (by character offset).
"""


def now():
    return datetime.datetime.now()


def _get_entity_filename(article_id):
    return article_id + "_gold.csv"


def _get_coref_filename(article_id):
    return article_id + "_coref.csv"


def create_training(wp_input, entity_def_input, training_output, limit=None):
    """ Create training for all Wikipedia articles, or a subset by defining `limit`"""
    wp_to_id = kb_creator.get_entity_to_id(entity_def_input)
    _process_wikipedia_texts(wp_input, wp_to_id, training_output, limit=limit)


def _process_wikipedia_texts(wikipedia_input, wp_to_id, training_output, limit=None):
    """
    Read the XML wikipedia data to parse out training data:
    raw text data + positive instances
    """
    title_regex = re.compile(r"(?<=<title>).*(?=</title>)")
    id_regex = re.compile(r"(?<=<id>)\d*(?=</id>)")

    read_ids = set()
    article_cnt = 0

    with bz2.open(wikipedia_input, mode="rb") as file:
        line = file.readline()
        article_text = ""
        article_title = None
        article_id = None
        reading_text = False
        reading_revision = False
        while line and (not limit or article_cnt < limit):
            clean_line = line.strip().decode("utf-8")

            if clean_line == "<revision>":
                reading_revision = True
            elif clean_line == "</revision>":
                reading_revision = False

            # Start reading new page
            if clean_line == "<page>":
                article_text = ""
                article_title = None
                article_id = None

            # finished reading this page
            elif clean_line == "</page>":
                if article_id:
                    try:
                        result = _process_wp_text(
                            wp_to_id,
                            article_id,
                            article_title,
                            article_text.strip(),
                            training_output,
                        )
                        if result:
                            article_cnt += 1
                            if article_cnt > 0 and article_cnt % 5000 == 0:
                                print(now(), "processed", article_cnt, "Wikipedia articles")
                    except Exception as e:
                        print("Error processing article", article_id, article_title, e)
                else:
                    print(
                        "Done processing a page, but couldn't find an article_id ?",
                        article_title,
                    )
                article_text = ""
                article_title = None
                article_id = None
                reading_text = False
                reading_revision = False

            # start reading text within a page
            if "<text" in clean_line:
                reading_text = True

            if reading_text:
                article_text += " " + clean_line

            # stop reading text within a page (we assume a new page doesn't start on the same line)
            if "</text" in clean_line:
                reading_text = False

            # read the ID of this article (outside the revision portion of the document)
            if not reading_revision:
                ids = id_regex.search(clean_line)
                if ids:
                    article_id = ids[0]
                    if article_id in read_ids:
                        print(
                            "Found duplicate article ID", article_id, clean_line
                        )  # This should never happen ...
                    read_ids.add(article_id)

            # read the title of this article (outside the revision portion of the document)
            if not reading_revision:
                titles = title_regex.search(clean_line)
                if titles:
                    article_title = titles[0].strip()

            line = file.readline()

    print(now(), "processed", article_cnt, "Wikipedia articles")


text_regex = re.compile(r"(?<=<text xml:space=\"preserve\">).*(?=</text)")


def _process_wp_text(
    wp_to_id, article_id, article_title, article_text, training_output
):
    found_entities = False

    # ignore meta Wikipedia pages
    if article_title.startswith("Wikipedia:"):
        return 0

    # remove the text tags
    text = text_regex.search(article_text).group(0)

    # stop processing if this is a redirect page
    if text.startswith("#REDIRECT"):
        return 0

    # get the raw text without markup etc, keeping only interwiki links
    clean_text = _get_clean_wp_text(text)

    # read the text char by char to get the right offsets for the interwiki links
    final_text = ""
    open_read = 0
    reading_text = True
    reading_entity = False
    reading_mention = False
    reading_special_case = False
    entity_buffer = ""
    mention_buffer = ""

    entityfile_name = _get_entity_filename(article_id)
    entityfile_loc = training_output / entityfile_name
    with entityfile_loc.open("w", encoding="utf8") as entityfile:
        wrote_header = False
        for index, letter in enumerate(clean_text):
            if letter == "[":
                open_read += 1
            elif letter == "]":
                open_read -= 1
            elif letter == "|":
                if reading_text:
                    final_text += letter
                # switch from reading entity to mention in the [[entity|mention]] pattern
                elif reading_entity:
                    reading_text = False
                    reading_entity = False
                    reading_mention = True
                else:
                    reading_special_case = True
            else:
                if reading_entity:
                    entity_buffer += letter
                elif reading_mention:
                    mention_buffer += letter
                elif reading_text:
                    final_text += letter
                else:
                    raise ValueError(
                        "Not sure at point", clean_text[index - 2 : index + 2]
                    )

            if open_read > 2:
                reading_special_case = True

            if open_read == 2 and reading_text:
                reading_text = False
                reading_entity = True
                reading_mention = False

            # we just finished reading an entity
            if open_read == 0 and not reading_text:
                if "#" in entity_buffer or entity_buffer.startswith(":"):
                    reading_special_case = True
                # Ignore cases with nested structures like File: handles etc
                if not reading_special_case:
                    if not mention_buffer:
                        mention_buffer = entity_buffer
                    start = len(final_text)
                    end = start + len(mention_buffer)
                    qid = wp_to_id.get(entity_buffer, None)
                    if qid:
                        if not wrote_header:
                            _write_training_entity(
                                outputfile=entityfile,
                                article_id="article_id",
                                alias="alias",
                                entity="WD_id",
                                start="start",
                                end="end",
                            )
                            wrote_header = True

                        _write_training_entity(
                            outputfile=entityfile,
                            article_id=article_id,
                            alias=mention_buffer,
                            entity=qid,
                            start=start,
                            end=end,
                        )
                        found_entities = True
                    final_text += mention_buffer

                entity_buffer = ""
                mention_buffer = ""

                reading_text = True
                reading_entity = False
                reading_mention = False
                reading_special_case = False

    if found_entities:
        _write_training_article(
            article_id=article_id,
            clean_text=final_text,
            training_output=training_output,
        )
        return 1
    else:
        entityfile_loc.unlink()  # removing empty files
    return 0


info_regex = re.compile(r"{[^{]*?}")
htlm_regex = re.compile(r"&lt;!--[^-]*--&gt;")
category_regex = re.compile(r"\[\[Category:[^\[]*]]")
file_regex = re.compile(r"\[\[File:[^[\]]+]]")
ref_regex = re.compile(r"&lt;ref.*?&gt;")  # non-greedy
ref_2_regex = re.compile(r"&lt;/ref.*?&gt;")  # non-greedy


def _get_clean_wp_text(article_text):
    clean_text = article_text.strip()

    # remove bolding & italic markup
    clean_text = clean_text.replace("'''", "")
    clean_text = clean_text.replace("''", "")

    # remove nested {{info}} statements by removing the inner/smallest ones first and iterating
    try_again = True
    previous_length = len(clean_text)
    while try_again:
        clean_text = info_regex.sub(
            "", clean_text
        )  # non-greedy match excluding a nested {
        if len(clean_text) < previous_length:
            try_again = True
        else:
            try_again = False
        previous_length = len(clean_text)

    # remove HTML comments
    clean_text = htlm_regex.sub("", clean_text)

    # remove Category and File statements
    clean_text = category_regex.sub("", clean_text)
    clean_text = file_regex.sub("", clean_text)

    # remove multiple =
    while "==" in clean_text:
        clean_text = clean_text.replace("==", "=")

    clean_text = clean_text.replace(". =", ".")
    clean_text = clean_text.replace(" = ", ". ")
    clean_text = clean_text.replace("= ", ".")
    clean_text = clean_text.replace(" =", "")

    # remove refs (non-greedy match)
    clean_text = ref_regex.sub("", clean_text)
    clean_text = ref_2_regex.sub("", clean_text)

    # remove additional wikiformatting
    clean_text = re.sub(r"&lt;blockquote&gt;", "", clean_text)
    clean_text = re.sub(r"&lt;/blockquote&gt;", "", clean_text)

    # change special characters back to normal ones
    clean_text = clean_text.replace(r"&lt;", "<")
    clean_text = clean_text.replace(r"&gt;", ">")
    clean_text = clean_text.replace(r"&quot;", '"')
    clean_text = clean_text.replace(r"&amp;nbsp;", " ")
    clean_text = clean_text.replace(r"&amp;", "&")

    # remove multiple spaces
    while "  " in clean_text:
        clean_text = clean_text.replace("  ", " ")

    return clean_text.strip()


def _write_training_article(article_id, clean_text, training_output):
    file_loc = training_output / "{}.txt".format(article_id)
    with file_loc.open("w", encoding="utf8") as outputfile:
        outputfile.write(clean_text)


def _write_training_entity(outputfile, article_id, alias, entity, start, end):
    line = "{}|{}|{}|{}|{}\n".format(article_id, alias, entity, start, end)
    outputfile.write(line)


def _write_coref_(outputfile, article_id, coref_id, start, end):
    line = "{}|{}|{}|{}\n".format(article_id, coref_id, start, end)
    outputfile.write(line)


def is_dev(article_id):
    return article_id.endswith("3")


def read_training(nlp, training_dir, dev, limit, kb=None, sentence=False, coref=False):
    """ This method provides training examples that correspond to the entity annotations found by the nlp object.
     When kb is provided (for training), it will include negative training examples by using the candidate generator,
     and it will only keep positive training examples that can be found in the KB.
     When kb=None (for testing), it will include all positive examples only."""
    data = []
    coref_data = dict()
    total_articles = 0
    total_entities = 0

    for textfile in training_dir.iterdir():
        if not limit or total_articles < limit:
            if textfile.name.endswith(".txt"):
                article_id = textfile.name.split(".")[0]
                if dev == is_dev(article_id):
                    with textfile.open("r", encoding="utf8") as f:
                        current_doc = None
                        try:
                            text = f.read()
                            # at this point we only need the NER and the sentence segmentation
                            other_pipes = ["entity_linker"]
                            # disable the parser if we don't need the sentence segmentation
                            if not sentence:
                                other_pipes = [
                                    pipe for pipe in nlp.pipe_names if pipe != "ner"
                                ]
                            current_doc = nlp(text, disable=other_pipes)
                            current_doc.user_data["orig_article_id"] = article_id
                        except Exception as e:
                            print("Problem parsing article", article_id, e)

                    if current_doc:
                        wp_entity_offsets = list()
                        wp_aliases = list()
                        wp_ids = list()

                        entityfile_name = _get_entity_filename(article_id)
                        entityfile_loc = training_dir / entityfile_name
                        with entityfile_loc.open("r", encoding="utf8") as entityfile:
                            for line in entityfile:
                                fields = line.replace("\n", "").split(sep="|")
                                article_id = fields[0]
                                alias = fields[1]
                                wd_id = fields[2]
                                start = fields[3]
                                end = fields[4]

                                if article_id != "article_id":
                                    offset = "{}_{}".format(start, end)
                                    wp_entity_offsets.append(offset)
                                    wp_aliases.append(alias)
                                    wp_ids.append(wd_id)
                        if sentence:
                            gold_data = _process_per_sentence(kb, current_doc, wp_entity_offsets, wp_aliases, wp_ids)
                        else:
                            gold_data = _read_per_article(kb, current_doc, wp_entity_offsets, wp_aliases, wp_ids)
                        total_articles += 1
                        total_entities += len(gold_data)
                        data.extend(gold_data)
                        if len(data) % 2500 == 0:
                            print(" -read", total_entities, "articles")
                        if coref:
                            entities_by_coref_id = _read_coreference(training_dir, current_doc)
                            coref_data[article_id] = (current_doc, entities_by_coref_id)

    print(" -read", total_articles, "articles with", total_entities, "entities")
    if coref:
        return data, coref_data
    return data


def _read_per_article(kb, article_doc, wp_entity_offsets, wp_aliases, wp_ids):
    """ Process a doc and create the gold links necessary for training & evaluation """
    ents_by_offset = dict()
    for ent in article_doc.ents:
        offset = "{}_{}".format(ent.start_char, ent.end_char)
        ents_by_offset[offset] = ent

    gold_entities = {}

    for offset, alias, wd_id in zip(wp_entity_offsets, wp_aliases, wp_ids):
        start, end = offset.split("_")
        found_ent = ents_by_offset.get(offset, None)
        if found_ent:
            if found_ent.text == alias:
                for ent in article_doc.ents:
                    entry = (ent.start_char, ent.end_char)
                    gold_entry = (int(start), int(end))
                    if entry == gold_entry:
                        # add both pos and neg examples (in random order)
                        # this will exclude examples not in the KB
                        if kb:
                            value_by_id = {}
                            candidates = kb.get_candidates(alias)
                            candidate_ids = [c.entity_ for c in candidates]
                            random.shuffle(candidate_ids)
                            for kb_id in candidate_ids:
                                if kb_id != wd_id:
                                    value_by_id[kb_id] = 0.0
                                else:
                                    value_by_id[kb_id] = 1.0
                            if value_by_id:
                                gold_entities[entry] = value_by_id
                        # if no KB, keep all positive examples
                        else:
                            gold_entities[entry] = {wd_id: 1.0}

    gold = GoldParse(doc=article_doc, links=gold_entities)
    return [(article_doc, gold)]


def _process_per_sentence(kb, article_doc, wp_entity_offsets, wp_aliases, wp_ids):
    """ Create a "doc" + gold links for each sentence in the original doc """
    ents_by_offset = dict()
    for ent in article_doc.ents:
        sent_length = len(ent.sent)
        # custom filtering to avoid too long or too short sentences
        if 5 < sent_length < 100:
            offset = "{}_{}".format(ent.start_char, ent.end_char)
            ents_by_offset[offset] = ent

    sent_by_start = dict()
    data_by_sent = dict()
    for offset, alias, wd_id in zip(wp_entity_offsets, wp_aliases, wp_ids):
        start, end = offset.split("_")
        found_ent = ents_by_offset.get(offset, None)
        if found_ent:
            if found_ent.text == alias:
                sent_start = found_ent.sent.start_char

                sent_doc = sent_by_start.get(sent_start, None)
                if not sent_doc:
                    sent_doc = found_ent.sent.as_doc()

                    # don't copy the user_data object, because then sent_offset will be the same across sent_doc's
                    sent_doc.user_data["orig_article_id"] = article_doc.user_data["orig_article_id"]
                    sent_doc.user_data["sent_offset"] = int(sent_start)
                    sent_by_start[sent_start] = sent_doc

                gold_start = int(start) - sent_start
                gold_end = int(end) - sent_start

                gold_entities = {}
                found_useful = False
                for ent in sent_doc.ents:
                    entry = (ent.start_char, ent.end_char)
                    gold_entry = (gold_start, gold_end)
                    if entry == gold_entry:
                        # add both pos and neg examples (in random order)
                        # this will exclude examples not in the KB
                        if kb:
                            value_by_id = {}
                            candidates = kb.get_candidates(alias)
                            candidate_ids = [c.entity_ for c in candidates]
                            random.shuffle(candidate_ids)
                            for kb_id in candidate_ids:
                                found_useful = True
                                if kb_id != wd_id:
                                    value_by_id[kb_id] = 0.0
                                else:
                                    value_by_id[kb_id] = 1.0
                            gold_entities[entry] = value_by_id
                        # if no KB, keep all positive examples
                        else:
                            found_useful = True
                            value_by_id = {wd_id: 1.0}

                            gold_entities[entry] = value_by_id
                    # currently feeding the gold data one entity per sentence at a time
                    # setting all other entities to empty gold dictionary
                    # else:
                    # gold_entities[entry] = {}
                if found_useful:
                    gold = data_by_sent.get(sent_doc, None)
                    if not gold:
                        gold = GoldParse(doc=sent_doc, links={})
                    gold.links.update(gold_entities)
                    data_by_sent[sent_doc] = gold

    article_data = []
    for s_doc, s_gold in data_by_sent.items():
        article_data.append((s_doc, s_gold))
    return article_data


def write_coreference_annotations(nlp, training_dir, parallelize=False):
    """
    Write coreference annotations for the textual dataset. For this functionality to work,
    make sure the nlp component has a neuralcoref pipeline component !
    """
    if parallelize:
        from dask import delayed
        from dask import compute

        list_written = list()
        for textfile in training_dir.iterdir():
            if textfile.name.endswith(".txt"):
                article_id = textfile.name.split(".")[0]
                with textfile.open("r", encoding="utf8") as f:
                    try:
                        text = f.read()
                        doc = nlp(text)
                        list_written.append(
                            delayed(_write_coreference_to_article)(doc, training_dir, textfile)
                        )
                    except Exception as e:
                        print("Problem parsing article", article_id, e)

        total_written = compute(sum(list_written))
        print("Written", total_written, "coref statements")
    else:
        total_written = 0
        for textfile in training_dir.iterdir():
            if textfile.name.endswith(".txt"):
                with textfile.open("r", encoding="utf8") as f:
                    article_id = textfile.name.split(".")[0]
                    try:
                        text = f.read()
                        doc = nlp(text)
                        total_written += _write_coreference_to_article(
                            doc, training_dir, textfile
                        )
                    except Exception as e:
                        print("Problem parsing article", article_id, e)

        print("Written", total_written, "coref clusters")


def _write_coreference_to_article(doc, training_dir, textfile):
    article_id = textfile.name.split(".")[0]
    written = 0

    coref_name = _get_coref_filename(article_id)
    coref_out_loc = training_dir / coref_name

    # write heading
    with coref_out_loc.open("w", encoding="utf8") as coreffile_out:
        _write_coref_(
            outputfile=coreffile_out,
            article_id="article_id",
            coref_id="coref_id",
            start="start",
            end="end",
        )

        # write each cluster, assigning artificial IDs for ease of parsing
        c_id = 0
        for cluster in doc._.coref_clusters:
            c_id += 1
            written += 1
            for mention in cluster.mentions:
                _write_coref_(
                    outputfile=coreffile_out,
                    article_id=article_id,
                    coref_id=c_id,
                    start=mention.start,
                    end=mention.end,
                )

    return written


def _read_coreference(training_dir, article):
    coref_name = _get_coref_filename(article.user_data["orig_article_id"])
    coref_out_loc = training_dir / coref_name
    entities_by_coref_id = dict()
    with coref_out_loc.open("r", encoding="utf8") as coreffile_in:
        for line in coreffile_in:
            fields = line.replace("\n", "").split(sep="|")
            article_id = fields[0]
            coref_id = fields[1]

            if article_id != "article_id":
                start = int(fields[2])
                end = int(fields[3])

                coref_list = entities_by_coref_id.get(coref_id, list())
                coref_list.append(article[start:end])
                entities_by_coref_id[coref_id] = coref_list
    return entities_by_coref_id


def _add_from_coref(article_doc, wp_entity_offsets, wp_aliases, wp_ids):
    for ent in article_doc.ents:
        offset = "{}_{}".format(ent.start_char, ent.end_char)
        try:
            wp_index = wp_entity_offsets.index(offset)
        except ValueError:
            wp_index = -1
        if wp_index >= 0:
            wp_id = wp_ids[wp_index]
            if ent._.is_coref:
                for coref_ent in ent._.coref_cluster:
                    coref_offset = "{}_{}".format(
                        coref_ent.start_char, coref_ent.end_char
                    )
                    if coref_offset not in wp_entity_offsets:
                        wp_entity_offsets.append(coref_offset)
                        wp_aliases.append(coref_ent.text)
                        wp_ids.append(wp_id)

    return wp_entity_offsets, wp_aliases, wp_ids
